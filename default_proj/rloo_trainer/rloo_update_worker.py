"""Ray actor that applies policy-gradient updates for RLOO.

The orchestrator (`rloo.py`) samples responses and computes rewards, then
calls this worker with tokenized sequences to perform gradient updates.

This file is intentionally incomplete. Students are expected to implement
`update(...)` while reusing the data/model/sampling setup provided here.
"""

import os
import warnings
import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

@ray.remote(num_gpus=1)
class RLOOUpdateWorker:
    """Owns policy/ref models and optimizer state for RLOO updates."""
    def __init__(
        self, 
        model_path, 
        optimizer_path, 
        scheduler_path,
        tokenizer_path=None, 
        ref_model_path=None,
        batch_size=64,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        group_size=16, 
        entropy_coefficient=0.01, 
        kl_divergence_coefficient=0.0, 
        lr_schedule='constant',
        learning_rate=1e-5, 
        weight_decay=0.01, 
        warmup_ratio=0.0,
        num_training_steps=250,
    ):
        self.model_path = model_path
        self.ref_model_path = ref_model_path if ref_model_path is not None else model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.group_size = group_size
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for RLOO, got {self.group_size}")
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        if warmup_ratio > 0:
            raise NotImplementedError("Warmup ratio > 0 is not supported for constant learning rate schedule")
        self.num_training_steps = num_training_steps

    def tear_down(self):
        """Release model/optimizer objects and clear GPU memory."""
        import gc
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ref_model'):
            del self.ref_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_checkpoint_paths(self, model_path, optimizer_path, scheduler_path, load_checkpoint=False):
        """Update output paths (and optionally reload state immediately)."""
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        if load_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load policy model, optional reference model, and optimizer/scheduler."""
        self.tear_down()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")
        self.model.gradient_checkpointing_enable()

        if self.kl_divergence_coefficient > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.ref_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device="cuda")
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        if self.optimizer_path and self.scheduler_path and os.path.exists(self.optimizer_path) and os.path.exists(self.scheduler_path):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")
            
            self.scheduler.load_state_dict(torch.load(self.scheduler_path))
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")

        self.model.train()

    def save_checkpoint(self):
        """Persist optimizer/scheduler state plus model+tokenizer weights."""
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        torch.save(self.scheduler.state_dict(), self.scheduler_path)

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)


    def update_gradient_accumulation(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        device='cuda',
    ):
        """Split incoming batch into microbatches and call `update(...)`."""
        update_metrics = None
        if self.gradient_accumulation_steps > 1:
            curr_batch_size = input_ids.shape[0]
            assert curr_batch_size % self.gradient_accumulation_steps == 0, (
                f"Flattened batch size {curr_batch_size} must be divisible by gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps}."
            )
            group_per_gradient_accumulation_step = curr_batch_size // self.gradient_accumulation_steps
            # Ensure each microbatch still contains full RLOO groups so the baseline is meaningful
            assert group_per_gradient_accumulation_step % self.group_size == 0, (
                f"Microbatch size {group_per_gradient_accumulation_step} must be divisible by group_size {self.group_size} "
                f"when using gradient_accumulation_steps={self.gradient_accumulation_steps}."
            )
            all_metrics = []
            for i in range(self.gradient_accumulation_steps):
                curr_input_ids = input_ids[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_attention_mask = attention_mask[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_is_response_token = is_response_token[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_rewards = rewards[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_sample_log_probs = None
                if sample_log_probs is not None:
                    curr_sample_log_probs = sample_log_probs[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                
                is_update_step = (i == self.gradient_accumulation_steps - 1)
                curr_update_metrics = self.update(
                    curr_input_ids,
                    curr_attention_mask,
                    curr_is_response_token,
                    curr_rewards,
                    curr_sample_log_probs,
                    is_update_step,
                    device,
                )
                all_metrics.append(curr_update_metrics)
            update_metrics = {}
            for metric_name in all_metrics[0].keys():
                update_metrics[metric_name] = np.mean([metric[metric_name] for metric in all_metrics]).item()
        else:
            update_metrics = self.update(
                input_ids,
                attention_mask,
                is_response_token,
                rewards,
                sample_log_probs,
                True,
                device,
            )

        return update_metrics

    # `is_update_step` is False on intermediate microbatches so we can
    # accumulate gradients before stepping optimizer/scheduler.
    def update(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        is_update_step: bool = True,
        device='cuda',
    ):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
        is_response_token = torch.tensor(is_response_token, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        if sample_log_probs is not None:
            sample_log_probs = torch.tensor(sample_log_probs, dtype=torch.float32, device=device)

        # run the policy forward pass and get per-token log-probs for each response token
        outputs = self.model(input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        response_mask = (attention_mask[:, 1:].bool() & is_response_token[:, 1:].bool())

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * response_mask.float()
        sequence_log_probs = token_log_probs.sum(dim=-1)

        # correct for the vLLM/HuggingFace numerical gap using importance weights
        if sample_log_probs is not None:
            importance_weights = (sequence_log_probs - sample_log_probs).exp().clamp(max=10.0).detach()
        else:
            importance_weights = torch.ones(sequence_log_probs.shape[0], device=device)
        importance_weight_mean = importance_weights.mean()

        # leave-one-out baseline: for each response, average the rewards of all other responses
        # in the same group, then subtract to get the advantage
        num_prompts = rewards.shape[0] // self.group_size
        rewards_grouped = rewards.view(num_prompts, self.group_size)
        baseline = (rewards_grouped.sum(dim=1, keepdim=True) - rewards_grouped) / (self.group_size - 1)
        advantages = (rewards_grouped - baseline).view(-1)

        pg_loss = -(importance_weights * advantages * sequence_log_probs).mean() / self.gradient_accumulation_steps
        total_loss = pg_loss

        # entropy bonus keeps the policy from collapsing to deterministic outputs
        entropy_loss = torch.tensor(0.0, device=device)
        if self.entropy_coefficient > 0:
            probs = F.softmax(shift_logits, dim=-1)
            entropy_per_token = -(probs * log_probs).sum(dim=-1) * response_mask.float()
            mean_entropy = entropy_per_token.sum() / response_mask.float().sum().clamp(min=1)
            entropy_loss = -self.entropy_coefficient * mean_entropy / self.gradient_accumulation_steps
            total_loss = total_loss + entropy_loss

        # KL penalty keeps the policy anchored to the SFT reference model
        kl_loss = torch.tensor(0.0, device=device)
        if self.kl_divergence_coefficient > 0:
            with torch.no_grad():
                ref_shift_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
                ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
            policy_probs = F.softmax(shift_logits, dim=-1)
            kl_per_token = (policy_probs * (log_probs - ref_log_probs)).sum(dim=-1) * response_mask.float()
            mean_kl = kl_per_token.sum() / response_mask.float().sum().clamp(min=1)
            kl_loss = self.kl_divergence_coefficient * mean_kl / self.gradient_accumulation_steps
            total_loss = total_loss + kl_loss

        total_loss.backward()

        if is_update_step:
            if self.gradient_clipping > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {
            "rloo_loss": pg_loss.item() * self.gradient_accumulation_steps,
            "importance_weight_mean": importance_weight_mean.item(),
            "kl_loss": kl_loss.item() * self.gradient_accumulation_steps,
            "rollout_accuracy": rewards.mean().item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
