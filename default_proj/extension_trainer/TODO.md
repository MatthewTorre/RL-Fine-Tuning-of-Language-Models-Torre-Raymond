# Project Extension TODO: Elo-Curriculum RLOO for Countdown

Goal: modify only the RLOO stage so prompts are sampled by an Elo-based curriculum instead of uniformly, while keeping the Countdown verifier, rollout generation, and RLOO update loss unchanged.

## Core Design Decisions Still Needed

- [ ] Decide the minimum viable extension scope:
  - Option A: Elo-rated curriculum sampler only, with no learned teacher.
  - Option B: Elo sampler plus a small adaptive "teacher" policy that tunes the sampling target over time.
  - Recommendation: implement Option A first, then add teacher/adaptive ablations only if time permits.
- [ ] Decide what a "problem rating" represents:
  - Per individual dataset row.
  - Per problem family / bucket, such as number count, target range, or initial estimated difficulty.
  - Recommendation: start per-row because it is easiest to update after each rollout batch.
- [ ] Decide what an "agent rating" represents:
  - One global scalar rating for the current policy checkpoint.
  - A moving rating per epoch / global step.
  - Recommendation: one global scalar rating updated online throughout RLOO.
- [ ] Decide success definition for Elo:
  - Strict success: `compute_score(...) == 1.0`.
  - Partial success: treat format reward `0.1` as a draw or fractional score.
  - Recommendation: start strict binary success, as proposed in `extension.tex`.
- [ ] Decide whether Elo updates happen per sampled response or per prompt group:
  - Per response: each of `group_size` completions independently updates ratings.
  - Per prompt group: update once using whether any sampled completion solved the prompt.
  - Recommendation: per prompt group is less noisy and aligns with pass@k; per response is closer to rollout-level RL signal.
- [ ] Decide initial problem ratings:
  - All problems start at 1500.
  - Bootstrap ratings from SFT checkpoint rollouts.
  - Use heuristic difficulty from problem metadata (target, number count, operations).
  - Recommendation: all 1500 for the first working version; bootstrap as an ablation.
- [ ] Decide Elo K-factor:
  - Constant K, e.g. 16, 24, or 32.
  - Decaying K over training.
  - Separate `K_agent` and `K_problem`.
  - Recommendation: constant `K=32` initially, then tune.
- [ ] Decide sampling distribution:
  - Gaussian over problem Elo centered near current agent Elo.
  - Gaussian centered slightly above current agent Elo to push difficulty.
  - Mixture of curriculum and uniform sampling for exploration.
  - Recommendation: mixture sampler: `p = (1 - epsilon) * gaussian_curriculum + epsilon * uniform`.
- [ ] Decide curriculum target offset:
  - Center at `R_agent`.
  - Center at `R_agent + delta`, e.g. +50 or +100.
  - Adaptive delta based on recent success rate.
  - Recommendation: start with fixed `delta=0` or `+50`; log enough metrics to tune.
- [ ] Decide Gaussian width:
  - Narrow width focuses on near-skill problems.
  - Wide width preserves diversity.
  - Recommendation: expose `elo_sigma` as a CLI arg and try 100, 200, 400.
- [ ] Decide evaluation protocol:
  - Compare against uniform RLOO under same number of environment interactions.
  - Use same initial SFT or IPO checkpoint.
  - Keep generation settings fixed across runs.
  - Report pass@k, rollout reward, exact correctness, and training curves.

## Files To Change

- [ ] `default_proj/rloo_trainer/rloo_dataset.py`
  - Add stable example IDs to dataset items.
  - Return `{"id": idx, "prompt": prompt, "ground_truth": ground_truth}` from `__getitem__`.
  - Update `collate_fn` to return `ids` alongside prompts and ground truths.
  - Add support for `shuffle=False` or a curriculum sampler path so RLOO does not always depend on uniform dataloader shuffling.
- [ ] `default_proj/rloo_trainer/rloo.py`
  - Add curriculum-related CLI args and trainer fields.
  - Initialize Elo state after loading the train dataset.
  - Replace or bypass the current `for train_iter, batch in enumerate(self.train_dataloader)` uniform loop when curriculum is enabled.
  - Sample prompt IDs from the Elo curriculum each global step.
  - After rewards are computed, update agent/problem Elo ratings.
  - Log curriculum metrics to W&B.
- [ ] New helper module, likely `default_proj/rloo_trainer/elo_curriculum.py`
  - Implement Elo expected score.
  - Implement rating updates.
  - Implement Gaussian/mixed curriculum sampling over problem ratings.
  - Implement serialization for ratings and curriculum state.
- [ ] `default_proj/rloo_trainer/train_rloo.sh` and `train_rloo_modal.sh`
  - Add env vars / command-line args for enabling curriculum runs.
  - Add experiment names that clearly distinguish uniform baseline vs Elo curriculum variants.
- [ ] Evaluation scripts / notebooks
  - Ensure pass@k evaluation can compare SFT, IPO, uniform RLOO, and Elo-curriculum RLOO checkpoints under the same settings.
  - Save result JSONs with clear names for report tables.

## Implementation Plan

### 1. Preserve a clean uniform RLOO baseline

- [ ] Run or confirm a default RLOO baseline before changing sampling.
- [ ] Record baseline settings:
  - `model_name`
  - `dataset_name`
  - `num_training_steps`
  - `batch_size`
  - `group_size`
  - temperature / top-p / top-k / max tokens
  - entropy coefficient
  - KL coefficient
  - learning rate
- [ ] Decide whether the extension starts from:
  - SFT checkpoint only.
  - IPO checkpoint.
  - Both, if time permits.
- [ ] Use the same initial checkpoint and training budget for uniform and curriculum comparisons.

### 2. Add stable dataset IDs

- [ ] Modify `RLOODataset.__getitem__` to include the dataset row index.
- [ ] Modify `collate_fn` to return `ids`.
- [ ] Confirm `RLOOTrainer.train()` can still run unchanged when curriculum is disabled.
- [ ] Add a quick debug log showing prompt IDs sampled per step.

Why this matters: Elo problem ratings need a persistent key. Dataset row index is the simplest stable key as long as the dataset split and preprocessing stay fixed.

### 3. Build the Elo state object

- [ ] Create a small class or dataclass for curriculum state:
  - `agent_rating: float`
  - `problem_ratings: np.ndarray`
  - `problem_attempts: np.ndarray`
  - `problem_successes: np.ndarray`
  - optional recent success history
- [ ] Implement expected agent success:

```python
expected = 1.0 / (1.0 + 10.0 ** ((problem_rating - agent_rating) / 400.0))
```

- [ ] Implement update for a single problem outcome:

```python
agent_delta = k_factor * (success - expected)
agent_rating += agent_delta
problem_rating -= agent_delta
```

- [ ] Decide whether to update `agent_rating` after each rollout or once from the mean batch delta.
- [ ] Add clipping or sanity bounds only if needed, e.g. keep ratings within `[500, 3000]`.
- [ ] Write a simple unit/smoke test mentally or in a tiny script:
  - If model solves a high-rated problem, agent rating increases more.
  - If model fails an easy problem, agent rating decreases more.
  - Problem rating moves opposite the agent rating.

### 4. Implement curriculum sampling

- [ ] Compute target difficulty:

```python
target_rating = agent_rating + elo_target_offset
```

- [ ] Convert problem ratings to Gaussian weights:

```python
weights = np.exp(-0.5 * ((problem_ratings - target_rating) / elo_sigma) ** 2)
```

- [ ] Mask or downweight recently repeated prompts if needed.
- [ ] Mix with uniform exploration:

```python
weights = (1 - epsilon) * weights / weights.sum() + epsilon / num_problems
```

- [ ] Sample `batch_size` problem IDs without replacement when possible.
- [ ] Fetch prompts and ground truths for those IDs.
- [ ] Return a batch shaped like the existing dataloader batch:

```python
{
    "ids": ids,
    "prompt": prompts,
    "ground_truth": ground_truths,
}
```

- [ ] Decide whether replacement is allowed:
  - Without replacement inside a batch.
  - With replacement across steps.
  - Recommendation: no replacement inside batch, allow repeats across steps based on weights.

### 5. Integrate sampler into `RLOOTrainer.train()`

- [ ] Add CLI args:
  - `--curriculum_type` with values like `uniform`, `elo_gaussian`.
  - `--elo_initial_rating` default `1500`.
  - `--elo_k_factor` default `32`.
  - `--elo_sigma` default `200`.
  - `--elo_target_offset` default `0` or `50`.
  - `--elo_uniform_mix` default `0.05` or `0.10`.
  - `--elo_success_mode` default `group_any`.
  - `--elo_bootstrap_path` optional.
  - `--save_curriculum_state` boolean or implicit with checkpoints.
- [ ] If `curriculum_type == uniform`, keep current dataloader behavior.
- [ ] If `curriculum_type == elo_gaussian`, replace the dataloader batch with `curriculum.sample_batch(batch_size)`.
- [ ] Keep the rest of the flow unchanged:
  - sampling responses with `SamplingWorker`
  - computing verifier rewards with `compute_score`
  - tokenizing rollouts
  - updating via `RLOOUpdateWorker`
  - saving checkpoints
- [ ] After `all_rewards` is computed, call `curriculum.update(ids, all_rewards)`.
- [ ] Make sure curriculum updates happen before the next sampling step.

### 6. Decide and implement success aggregation

- [ ] For `group_any`:

```python
success = float(max(curr_rewards) == 1.0)
```

- [ ] For `group_mean_binary`:

```python
success = float(np.mean([r == 1.0 for r in curr_rewards]))
```

- [ ] For `per_response`:
  - Apply one Elo update per generated response.
  - This can move ratings faster and may need lower K.
- [ ] Log which success mode is used in W&B config.
- [ ] Recommendation: implement `group_any` first because RLOO samples `group_size` completions per prompt and final evaluation uses pass@k.

### 7. Save and resume curriculum state

- [ ] Save curriculum state next to model checkpoints:
  - `curriculum_state.json` or `curriculum_state.npz`
  - current `agent_rating`
  - `problem_ratings`
  - `problem_attempts`
  - `problem_successes`
  - curriculum hyperparameters
- [ ] Load curriculum state when resuming from a checkpoint.
- [ ] Include state in both latest checkpoint and periodic saved checkpoints.
- [ ] Ensure this does not affect normal uniform RLOO runs.

### 8. Add W&B logging

- [ ] Log scalar curriculum metrics each global step:
  - `curriculum/agent_rating`
  - `curriculum/target_rating`
  - `curriculum/sample_rating_mean`
  - `curriculum/sample_rating_std`
  - `curriculum/sample_rating_min`
  - `curriculum/sample_rating_max`
  - `curriculum/batch_success_rate`
  - `curriculum/recent_success_rate`
  - `curriculum/mean_problem_attempts`
  - `curriculum/unique_problem_fraction`
- [ ] Log histograms periodically:
  - sampled problem ratings
  - all problem ratings
  - attempts per problem
  - success rate per problem if enough data exists
- [ ] Add a W&B table with sampled prompts, target, numbers, ratings, rewards, and responses for qualitative inspection.

### 9. Optional SFT warm-start problem rating bootstrap

- [ ] Decide whether bootstrap is needed for the final extension.
- [ ] If yes, run the SFT checkpoint over each training prompt with a small number of samples.
- [ ] Convert empirical success rates to initial problem Elo ratings.
- [ ] Possible mapping:
  - High SFT success -> lower problem rating.
  - Low SFT success -> higher problem rating.
- [ ] Save bootstrap ratings to a file and pass via `--elo_bootstrap_path`.
- [ ] Keep all-1500 initialization as a baseline ablation because it is cleaner and easier to explain.

### 10. Optional adaptive teacher

- [ ] Decide whether to implement a separate teacher or only a heuristic scheduler.
- [ ] Simple teacher option:
  - Track recent success rate.
  - If success rate is high, increase `elo_target_offset`.
  - If success rate is low, decrease `elo_target_offset`.
- [ ] More complex teacher option:
  - Choose `target_rating` or `sigma` using bandit-style rewards based on student improvement.
- [ ] Recommendation: avoid a complex learned teacher unless the core Elo sampler is already working and evaluated.

## Experiment Plan

### Required baseline runs

- [ ] SFT checkpoint evaluation:
  - Already reported in `extension.tex` with pass@k.
  - Confirm artifact path and evaluation JSON.
- [ ] IPO checkpoint evaluation:
  - Train/fix IPO if needed.
  - Evaluate pass@k under the same prompt count and sampling settings.
- [ ] Uniform RLOO baseline:
  - Start from the same SFT or IPO checkpoint selected for curriculum.
  - Use default dataloader sampling.
  - Match total environment interactions against curriculum runs:

```text
env_interactions = num_training_steps * batch_size * group_size
```

### Curriculum runs

- [ ] Elo Gaussian curriculum, all problem ratings initialized to 1500.
- [ ] Elo Gaussian + uniform mixture (`epsilon` around 0.05 or 0.10).
- [ ] Optional: Elo Gaussian with SFT-bootstrap problem ratings.
- [ ] Optional: offset ablation (`target_offset = 0`, `+50`, `+100`).
- [ ] Optional: sigma ablation (`sigma = 100`, `200`, `400`).

### Metrics to report

- [ ] Final pass@k on held-out Countdown test prompts:
  - pass@1
  - pass@2
  - pass@4
  - pass@8
  - pass@16
- [ ] Sample efficiency:
  - verifier reward vs environment interactions
  - exact correctness vs environment interactions
  - pass@k vs environment interactions if periodic eval is feasible
- [ ] Training diagnostics:
  - RLOO policy-gradient loss
  - entropy
  - KL if enabled
  - importance weight mean / max
  - reward mean
- [ ] Curriculum diagnostics:
  - agent Elo over time
  - mean sampled problem Elo over time
  - distribution of problem ratings over time
  - success rate by problem Elo bucket

## Report / Analysis TODO

- [ ] Explain that the extension isolates data ordering:
  - same verifier
  - same reward
  - same RLOO update
  - changed prompt sampler only
- [ ] Describe why Countdown is suitable:
  - sparse reward
  - verifier provides objective correctness
  - problem difficulty varies naturally
- [ ] Include equations:
  - binary success conversion
  - Elo expected success
  - agent/problem rating updates
  - Gaussian sampling probability over ratings
- [ ] Include a small algorithm block:
  - initialize ratings
  - sample batch by curriculum
  - generate `group_size` responses
  - compute verifier rewards
  - update RLOO policy
  - update Elo ratings
- [ ] Include ablation table or figure:
  - uniform RLOO vs Elo curriculum
  - optionally sigma / offset / bootstrap variants
- [ ] Discuss failure modes:
  - ratings may be noisy early because model rarely solves prompts
  - all-1500 initialization gives no initial ordering until attempts accumulate
  - curriculum can overfocus on a narrow band of tasks
  - group-size pass/fail can inflate success relative to single-rollout ability
  - if the sampler becomes too hard too early, sparse reward returns
- [ ] Discuss why a uniform mixture helps:
  - prevents starvation of under-sampled problems
  - preserves exploration
  - gives ratings a chance to correct themselves

## Concrete Acceptance Criteria

- [ ] Uniform RLOO still runs with unchanged behavior when curriculum is disabled.
- [ ] Curriculum RLOO run samples prompt batches from Elo curriculum, not dataloader shuffle.
- [ ] Agent and problem ratings update after each sampled batch.
- [ ] Curriculum state is saved and reloadable.
- [ ] W&B contains both standard RLOO metrics and curriculum metrics.
- [ ] At least one curriculum run and one matched uniform baseline run are evaluated with pass@k.
- [ ] Final report can answer:
  - Did Elo curriculum improve sample efficiency?
  - Did it improve or match final pass@k?
  - Which design choices mattered most?

