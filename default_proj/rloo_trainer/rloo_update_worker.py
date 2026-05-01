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
        # TODO(student): implement one RLOO policy update.
        #
        # Objective (Eq. 10 in spec) -- RLOO policy gradient:
        #   (1/k) * sum_i [ R(y^i, x) - (1/(k-1)) * sum_{j != i} R(y^j, x) ]
        #                * grad log pi(y^i | x)
        # where y^1,...,y^k ~ pi_theta(.|x) are k=group_size responses per prompt.
        #
        # With importance weighting (Eq. 11) since vLLM and HF may differ:
        #   w(y, x) = exp( log pi_theta(y|x) - log mu(y|x) )  clipped to max value
        # Multiply each advantage by its importance weight before forming the loss.
        #
        # -----------------------------------------------------------------------
        # SETUP
        # -----------------------------------------------------------------------
        # Inputs arrive as numpy arrays of shape [batch_size * group_size, seq_len].
        # Convert all numpy arrays to torch tensors and move to `device`.
        #   input_ids         (N, L)  where N = batch_size * group_size
        #   attention_mask    (N, L)
        #   is_response_token (N, L)  1=response, 0=prompt
        #   rewards           (N,)    scalar reward per response
        #   sample_log_probs  (N,)    log mu(y|x) from vLLM sampling
        #
        # STEP 1 -- FORWARD PASS & PER-TOKEN LOG-PROBS
        #   logits = model(input_ids, attention_mask).logits     shape (N, L, vocab)
        #   Shift by 1: shift_logits (N, L-1, vocab), shift_labels (N, L-1), shift_mask (N, L-1).
        #   log_probs_per_token = F.log_softmax(shift_logits, dim=-1) gathered at shift_labels.
        #   Apply shift_mask; sum over response tokens -> sequence_logps  shape (N,)
        #
        # STEP 2 -- IMPORTANCE WEIGHTS  (Eq. 11)
        #   log_ratio = sequence_logps - sample_log_probs    shape (N,)
        #   importance_weights = log_ratio.exp().clamp(max=some_clip_value)
        #   importance_weight_mean = importance_weights.mean()  <- log this metric
        #
        # STEP 3 -- LEAVE-ONE-OUT BASELINE  (Eq. 10)
        #   Reshape rewards to (batch_size_prompts, group_size).
        #   For each sample i in a group, baseline_i = mean of rewards of all j != i.
        #     baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (group_size - 1)
        #   advantages = rewards - baseline    shape (batch_size_prompts, group_size)
        #   Flatten back to (N,).
        #
        # STEP 4 -- POLICY GRADIENT LOSS
        #   pg_loss = -mean( importance_weights * advantages * sequence_logps )
        #   Scale by (1 / gradient_accumulation_steps).
        #
        # STEP 5 -- ENTROPY REGULARIZATION (optional, if entropy_coefficient > 0)
        #   token_probs = shift_logits.softmax(dim=-1)
        #   entropy_per_token = -(token_probs * log_probs_all_tokens).sum(dim=-1)
        #   Apply shift_mask; average over response tokens -> mean_entropy
        #   entropy_loss = -entropy_coefficient * mean_entropy
        #   Add entropy_loss to total loss.
        #
        # STEP 6 -- KL PENALTY (optional, if kl_divergence_coefficient > 0)
        #   with torch.no_grad(): ref_logits = ref_model(input_ids, attention_mask).logits
        #   kl_per_token = F.kl_div(policy_log_probs, ref_probs, reduction='none').sum(-1)
        #   Apply shift_mask; average -> mean_kl
        #   kl_loss = kl_divergence_coefficient * mean_kl
        #   Add kl_loss to total loss.
        #   Log kl_loss as a metric.
        #
        # STEP 7 -- BACKWARD
        #   total_loss.backward()
        #
        # STEP 8 -- OPTIMIZER STEP (only if is_update_step)
        #   if is_update_step:
        #       torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        #       optimizer.step()
        #       scheduler.step()
        #       optimizer.zero_grad()
        #
        # -----------------------------------------------------------------------
        # RETURN METRICS DICT  (Section 4.3.2 required metrics)
        # -----------------------------------------------------------------------
        #   return {
        #       "rloo_loss":              float,
        #       "importance_weight_mean": float,   <- mean of w(y,x) across batch
        #       "kl_loss":                float,   <- 0.0 if kl_divergence_coefficient == 0
        #       "rollout_accuracy":       float,   <- mean(rewards) over the batch
        #       "lr":                     float,   <- scheduler.get_last_lr()[0]
        #   }
        raise NotImplementedError("This function is not implemented")
