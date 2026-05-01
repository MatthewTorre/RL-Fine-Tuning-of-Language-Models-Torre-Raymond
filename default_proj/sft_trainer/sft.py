"""Starter SFT training entrypoint for the class project.

This file is intentionally incomplete. Students are expected to implement
`train(...)` while reusing the data/model setup provided here.
"""

import sys
from pathlib import Path

# Allow `python sft_trainer/sft.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from sft_trainer.sft_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device='cuda', use_gradient_checkpointing=True):
    """Load policy model + tokenizer for SFT training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()
    return model, tokenizer

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def train(
    model, 
    tokenizer, 
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0
):
    #
    # Objective (Eq. 6 in spec):
    #   max_theta  E_{x,y in D}  sum_{t=1}^{|y|}  log pi_theta(y_t | x, y_{<t})
    # e.g,minimize cross-entropy loss but (only) on completion (response) tokens.
    #
    # Each batch dict has three tensors:
    #   input_ids         (B, prompt_len + response_len) -> move to device before passing to model
    #   attention_mask    (B, prompt_len + response_len)  1=real token, 0=pad
    #   is_response_token (B, prompt_len + response_len) -> 1=response token, 0=prompt token
    #
   

    # count epochs
    for epoch in range(num_epochs):
        #  gets batches from the dataloader for every epoch
        for batch_idx, batch in enumerate(train_dataloader):

            # for each batch dict, we have 3 different tensors
            # Move tensors to `device`.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            is_response_token = batch['is_response_token'].to(device)

            # step 1, forward pass
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)    
            logits = model_output.logits


            # autoregressive model so logits are used to predict the nexttoken
            response_pred_logits = logits[:, :-1, :]
            nt_labels = input_ids[:, 1:] # ground truth taken at each next position
            to_mask = is_response_token[:, 1:]

            # step 3; PER-TOKEN CROSS-ENTROPY
            #   Use F.cross_entropy(reduction='none') on flattened logits/labels.

            # reshaping results back to (B, L-1)
            B, L_minus_1, vocab_size = response_pred_logits.shape
            flat_logits = response_pred_logits.view(-1, vocab_size)   # (B*(L-1), vocab_size)
            flat_labels = nt_labels.view(-1)                          # (B*(L-1),)

            loss_per_token = F.cross_entropy(flat_logits, flat_labels, reduction='none')  
            loss_per_token = loss_per_token.view(response_pred_logits.size(0), -1)


            # step 4 MASK TO RESPONSE TOKENS ONLY & AVERAGE
            masked_loss = (loss_per_token * to_mask)  # prompt positions become 0

            # average and divide by the actual number of response token, microbatch loss must be scaled appropriately
            num_response_tokens = to_mask.sum()  # total number of response tokens in the batch
            loss = masked_loss.sum() /  num_response_tokens
            loss = loss/ gradient_accumulation_steps


            # step 5 TOKEN ACCURACY (metric, no gradient needed)
            with torch.no_grad():
                pred_tokens = response_pred_logits.argmax(dim=-1)
                truth = (pred_tokens == nt_labels)
                token_accuracy = (truth * to_mask).sum() / to_mask.sum()

            train_loss = loss.item()
            train_token_accuracy = token_accuracy.item()

            # step 6 BACKWARD
            loss.backward()

            # STEP 7: OPTIMIZER STEP (only on gradient accumulation boundary)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                # adjusts the lr according to schedule
                scheduler.step()

                # clear all accum grads back to zero for new microbatch
                optimizer.zero_grad()

        # EVALUATION (after each epoch)
        model.eval()
        test_losses = []
        test_accuracies = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                is_response_token = batch['is_response_token'].to(device)

                model_output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = model_output.logits

                response_pred_logits = logits[:, :-1, :]
                nt_labels = input_ids[:, 1:]
                to_mask = is_response_token[:, 1:]

                B, L_minus_1, vocab_size = response_pred_logits.shape
                flat_logits = response_pred_logits.view(-1, vocab_size)
                flat_labels = nt_labels.view(-1)

                loss_per_token = F.cross_entropy(flat_logits, flat_labels, reduction='none')
                loss_per_token = loss_per_token.view(response_pred_logits.size(0), -1)

                masked_loss = (loss_per_token * to_mask)
                num_response_tokens = to_mask.sum()
                loss = masked_loss.sum() / num_response_tokens

                pred_tokens = response_pred_logits.argmax(dim=-1)
                truth = (pred_tokens == nt_labels)
                token_accuracy = (truth * to_mask).sum() / to_mask.sum()

                test_losses.append(loss.item())
                test_accuracies.append(token_accuracy.item())

        model.train()

        # LOGGING to W&B  (Section 4.2.2 needs these required metrics)
        wandb.log({"train/loss": train_loss, "train/token_accuracy": train_token_accuracy})
        wandb.log({"test/loss": sum(test_losses) / len(test_losses), "test/token_accuracy": sum(test_accuracies) / len(test_accuracies)})

        # check pointing
        if save_model:
            save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='Asap7772/cog_behav_all_strategies')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model, 
        tokenizer, 
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping
    )

if __name__ == "__main__":
    main()
