"""Starter IPO training entrypoint for the class project.

This script wires model loading, data loading, and optimizer setup.
Students are expected to implement `train(...)` for the IPO objective.
"""

import sys
from pathlib import Path

# Allow `python ipo_trainer/ipo.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from ipo_trainer.ipo_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
import copy
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device, use_gradient_checkpointing=True):
    """Load trainable policy model and frozen reference model."""
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

    # IPO compares policy preferences to a fixed baseline policy.
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    return model, tokenizer, reference_model

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

# HELPER -- compute_sequence_logps(model, input_ids, attention_mask, is_response_token):
    #   Run forward pass -> logits (B, L, vocab).
    #   Shift by 1: shift_logits (B, L-1, vocab), shift_labels (B, L-1), shift_mask (B, L-1).
    #   Per-token log-probs = F.log_softmax(shift_logits, dim=-1) gathered at shift_labels.
    #   Mask to response tokens; sum (or average if average_logps=True) -> (B,) sequence logps.
def compute_sequence_logps(model, input_ids, attention_mask, is_response_token, average_logps=False):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn   = attention_mask[:, 1:]
    shift_resp   = is_response_token[:, 1:]

    # Only count positions that are not padding and are response tokens
    mask = (shift_attn.bool() & shift_resp.bool())

    log_probs = F.log_softmax(shift_logits, dim=-1)
    tok_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    tok_logps = tok_logps * mask.to(tok_logps.dtype)
    seq_logps = tok_logps.sum(dim=-1)

    if average_logps:
        denom = mask.sum(dim=-1).clamp_min(1).to(seq_logps.dtype)
        seq_logps = seq_logps / denom

    return seq_logps

def train(
    model, 
    tokenizer, 
    reference_model,
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0,
    beta=0.1,
    average_logps=False,
    loss_type='ipo',
):
    model.to(device)
    reference_model.to(device)
    reference_model.eval()
    test_iter = iter(test_dataloader)
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids_w = batch["input_ids_w"].to(device)
            input_ids_l = batch["input_ids_l"].to(device)
            attention_mask_w = batch["attention_mask_w"].to(device)
            attention_mask_l = batch["attention_mask_l"].to(device)
            is_response_token_w = batch["is_response_token_w"].to(device)
            is_response_token_l = batch["is_response_token_l"].to(device)

            # STEP 1 -- POLICY LOG-PROBS (with gradients)
            # logps_policy_w = compute_sequence_logps(model, input_ids_w, attention_mask_w, is_response_token_w)
            # logps_policy_l = compute_sequence_logps(model, input_ids_l, attention_mask_l, is_response_token_l)
            logps_policy_w = compute_sequence_logps(model, input_ids_w, attention_mask_w, is_response_token_w, average_logps=average_logps)
            logps_policy_l = compute_sequence_logps(model, input_ids_l, attention_mask_l, is_response_token_l, average_logps=average_logps)

            # STEP 2 -- REFERENCE LOG-PROBS (no gradients, frozen model)
            #   with torch.no_grad():
            #       logps_ref_w = compute_sequence_logps(reference_model, input_ids_w, ...)
            #       logps_ref_l = compute_sequence_logps(reference_model, input_ids_l, ...)
            
            with torch.no_grad():
                logps_ref_w = compute_sequence_logps(reference_model, input_ids_w, attention_mask_w, is_response_token_w, average_logps=average_logps)
                logps_ref_l = compute_sequence_logps(reference_model, input_ids_l, attention_mask_l, is_response_token_l, average_logps=average_logps)
                

            # STEP 3 -- IMPLICIT REWARD MARGIN  h  (Eq. 9)
            #   h = (logps_policy_w - logps_ref_w) - (logps_policy_l - logps_ref_l)
            #   reward_margin = h.mean()   <- log this as "IPO reward margin"
            h = (logps_policy_w - logps_ref_w) - (logps_policy_l - logps_ref_l)
            train_reward_margin = h.mean()

            # STEP 4 -- IPO LOSS
            #   ipo_loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
            #   Scale by (1 / gradient_accumulation_steps) before backward.
            ipo_loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
            ipo_loss = ipo_loss / gradient_accumulation_steps
            
            # STEP 5 -- BACKWARD
            #   ipo_loss.backward()
            ipo_loss.backward()

            # STEP 6 -- OPTIMIZER STEP (on gradient accumulation boundary)
            #   torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            #   optimizer.step() ; scheduler.step() ; optimizer.zero_grad()
            if (step + 1) % gradient_accumulation_steps == 0:
                if (gradient_clipping > 0.0):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            wandb.log({"train/ipo_loss": ipo_loss.item(), "train/reward_margin": train_reward_margin.item()})
        
            # -----------------------------------------------------------------------
            # EVALUATION (after each epoch on test_dataloader)
            # -----------------------------------------------------------------------
            #   model.eval() ; torch.no_grad()
            #   Repeat steps 1-4, accumulate test IPO loss and test reward margin.
            #   model.train()
            model.eval()
            with torch.no_grad():
                try:
                    batch = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_dataloader)   # restart when exhausted
                    batch = next(test_iter)
                input_ids_w = batch["input_ids_w"].to(device)
                input_ids_l = batch["input_ids_l"].to(device)
                attention_mask_w = batch["attention_mask_w"].to(device)
                attention_mask_l = batch["attention_mask_l"].to(device)
                is_response_token_w = batch["is_response_token_w"].to(device)
                is_response_token_l = batch["is_response_token_l"].to(device)

                # STEP 1 -- POLICY LOG-PROBS (with gradients)
                # logps_policy_w = compute_sequence_logps(model, input_ids_w, attention_mask_w, is_response_token_w)
                # logps_policy_l = compute_sequence_logps(model, input_ids_l, attention_mask_l, is_response_token_l)
                logps_policy_w = compute_sequence_logps(model, input_ids_w, attention_mask_w, is_response_token_w, average_logps=average_logps)
                logps_policy_l = compute_sequence_logps(model, input_ids_l, attention_mask_l, is_response_token_l, average_logps=average_logps)

                # STEP 2 -- REFERENCE LOG-PROBS (no gradients, frozen model)
                #   with torch.no_grad():
                #       logps_ref_w = compute_sequence_logps(reference_model, input_ids_w, ...)
                #       logps_ref_l = compute_sequence_logps(reference_model, input_ids_l, ...)
                logps_ref_w = compute_sequence_logps(reference_model, input_ids_w, attention_mask_w, is_response_token_w, average_logps=average_logps)
                logps_ref_l = compute_sequence_logps(reference_model, input_ids_l, attention_mask_l, is_response_token_l, average_logps=average_logps)
                
                # STEP 3 -- IMPLICIT REWARD MARGIN  h  (Eq. 9)
                #   h = (logps_policy_w - logps_ref_w) - (logps_policy_l - logps_ref_l)
                #   reward_margin = h.mean()   <- log this as "IPO reward margin"
                h = (logps_policy_w - logps_ref_w) - (logps_policy_l - logps_ref_l)
                test_reward_margin = h.mean()

                # STEP 4 -- IPO LOSS
                #   ipo_loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
                #   Scale by (1 / gradient_accumulation_steps) before backward.
                ipo_loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
                ipo_loss = ipo_loss / gradient_accumulation_steps

                # -----------------------------------------------------------------------
                # LOGGING to W&B  (Section 4.3.1 required metrics)
                # -----------------------------------------------------------------------
                #   wandb.log({"train/ipo_loss": ..., "train/reward_margin": ...})
                #   wandb.log({"test/ipo_loss":  ..., "test/reward_margin":  ...})
                # Training + test loss and reward margin gets logged every batch, per Ed recommendation
                # Test loss and reward margin only gets logged every epoch. 

        
                wandb.log({"test/ipo_loss":  ipo_loss.item(), "test/reward_margin":  test_reward_margin.item()})

        # -----------------------------------------------------------------------
        # CHECKPOINTING
        # -----------------------------------------------------------------------
        #   if save_model: call save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)
        if save_model: 
            save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)


    # TODO(student): implement IPO/DPO-style pairwise optimization.
    #
    # Objective (Eq. 9 in spec) -- IPO:
    #   L_IPO = || h^{yw,yl}_{pi_theta} - (2*beta)^{-1} ||_2^2
    #   where h = [log pi_theta(yw|x) - log pi_ref(yw|x)]
    #           - [log pi_theta(yl|x) - log pi_ref(yl|x)]
    #
    # The implicit reward for a single response (Eq. 7):
    #   r_theta(x,y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))
    # So h = (1/beta) * (r_theta(x,yw) - r_theta(x,yl))
    #
    # -----------------------------------------------------------------------
    # OUTER LOOP: for epoch in range(num_epochs): for batch in train_dataloader:
    # -----------------------------------------------------------------------
    #
    # Each batch dict has six tensors (from ipo_dataset.py collate_fn):
    #   input_ids_w / attention_mask_w / is_response_token_w   <- chosen
    #   input_ids_l / attention_mask_l / is_response_token_l   <- rejected
    #
    # HELPER -- compute_sequence_logps(model, input_ids, attention_mask, is_response_token):
    #   Run forward pass -> logits (B, L, vocab).
    #   Shift by 1: shift_logits (B, L-1, vocab), shift_labels (B, L-1), shift_mask (B, L-1).
    #   Per-token log-probs = F.log_softmax(shift_logits, dim=-1) gathered at shift_labels.
    #   Mask to response tokens; sum (or average if average_logps=True) -> (B,) sequence logps.
    #
    # STEP 1 -- POLICY LOG-PROBS (with gradients)
    #   logps_policy_w = compute_sequence_logps(model, input_ids_w, attention_mask_w, is_response_token_w)
    #   logps_policy_l = compute_sequence_logps(model, input_ids_l, attention_mask_l, is_response_token_l)
    #
    # STEP 2 -- REFERENCE LOG-PROBS (no gradients, frozen model)
    #   with torch.no_grad():
    #       logps_ref_w = compute_sequence_logps(reference_model, input_ids_w, ...)
    #       logps_ref_l = compute_sequence_logps(reference_model, input_ids_l, ...)
    #
    # STEP 3 -- IMPLICIT REWARD MARGIN  h  (Eq. 9)
    #   h = (logps_policy_w - logps_ref_w) - (logps_policy_l - logps_ref_l)
    #   reward_margin = h.mean()   <- log this as "IPO reward margin"
    #
    # STEP 4 -- IPO LOSS
    #   ipo_loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
    #   Scale by (1 / gradient_accumulation_steps) before backward.
    #
    # STEP 5 -- BACKWARD
    #   ipo_loss.backward()
    #
    # STEP 6 -- OPTIMIZER STEP (on gradient accumulation boundary)
    #   torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
    #   optimizer.step() ; scheduler.step() ; optimizer.zero_grad()
    #
    # -----------------------------------------------------------------------
    # EVALUATION (after each epoch on test_dataloader)
    # -----------------------------------------------------------------------
    #   model.eval() ; torch.no_grad()
    #   Repeat steps 1-4, accumulate test IPO loss and test reward margin.
    #   model.train()
    #
    # -----------------------------------------------------------------------
    # LOGGING to W&B  (Section 4.3.1 required metrics)
    # -----------------------------------------------------------------------
    #   wandb.log({"train/ipo_loss": ..., "train/reward_margin": ...})
    #   wandb.log({"test/ipo_loss":  ..., "test/reward_margin":  ...})
    #
    # -----------------------------------------------------------------------
    # CHECKPOINTING
    # -----------------------------------------------------------------------
    #   if save_model: call save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4-dpo')
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
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--average_logps', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='dpo')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer, reference_model = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

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
        reference_model,
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping,
        args.beta,
        args.average_logps,
        args.loss_type
    )

if __name__ == "__main__":
    main()
