Project Extension TODO: Elo-Curriculum RLOO for Countdown

Goal: modify only the RLOO stage so prompts are sampled by an Elo-based curriculum instead of uniformly, while keeping the Countdown verifier, rollout generation, and RLOO update loss unchanged.

## Chosen Design

- Implement a Gaussian-based Elo-rated curriculum sampler only, with no learned teacher.
- Each individual training dataset row has its own problem Elo rating.
- The agent has one moving Elo rating, updated every RLOO global step.
- Elo updates happen once per prompt group, after sampling `group_size` responses for that prompt.
- The agent's Elo score for a prompt should reflect both correctness and ease of solution.
- Evaluation remains exactly the same as `rloo_trainer`; only training prompt selection changes.
- Checkpointing follows the same pattern as RLOO training, with curriculum state saved alongside the latest/model checkpoints.

## Group Success-Fraction Elo Score

Use the fraction of `group_size` sampled responses that solve the prompt. This treats the sampled group as parallel rollouts, not as an ordered sequence of attempts.

- If all sampled responses solve the prompt, set `S_A = 1.0`.
- If half of the sampled responses solve the prompt, set `S_A = 0.5`.
- If one out of `group_size` sampled responses solves the prompt, set `S_A = 1 / group_size`.
- If none of the sampled responses solve the prompt, set `S_A = 0.0`.
- The problem score is `S_P = 1 - S_A`.

Implementation sketch:

```python
def success_fraction(group_rewards):
    return sum(reward == 1.0 for reward in group_rewards) / len(group_rewards)
```

This makes Elo gain/loss proportional to ease without relying on arbitrary response ordering. If the policy solves many independent samples for the same prompt, the prompt is easy for the policy; if it solves only a few or none, the prompt is harder. This is a fractional Elo score rather than a binary win/loss, which is valid for Elo-style updates and gives more information than `any_success`.

## Elo Formulas

Expected success/ease score for agent `A` against problem `P`:

```text
E_A = 1 / (1 + 10 ** ((R_P - R_A) / 400))
E_P = 1 - E_A
```

Agent update:

```text
R_A' = R_A + K_agent * (S_A - E_A)
```

Problem update:

```text
R_P' = R_P + K_problem * (S_P - E_P)
     = R_P - K_problem * (S_A - E_A)
```

Use:

- `K_agent = 32`
- `K_problem = 800 / N`, where `N` is the number of times that problem has been sampled after incrementing the attempt count.

We intentionally allow large early problem-rating swings so the curriculum quickly creates a wide spread of problem Elos. Keep this behavior even if ratings move sharply at the beginning.

## Initial Agent Elo

The SFT baseline pass@k values are:

- pass@1: 28.6%
- pass@2: 44.1%
- pass@4: 59.6%
- pass@8: 72.0%
- pass@16: 78.0%

Because our Elo score uses group success fraction rather than strict pass@1, initialize the RLOO agent near the implied per-sample solve probability rather than pass@k success of at least one sample.

Approximate per-sample solve probability from the pass@k curve:

```text
S_fraction ~= 0.286
```

Using average problem Elo `1500`, the implied agent rating is:

```text
R_agent = 1500 + 400 * log10(S_fraction / (1 - S_fraction))
        ~= 1340
```

Decision:

- Initialize the RLOO agent Elo to `1340`.
- This uses the SFT baseline pass@1 as the best available estimate of single-sample success probability, which matches the group success-fraction Elo score.

## Initial Problem Elo Heuristic

Initialize problem ratings using empirical solve ease from: The SFT baseline model.

For each training problem:

1. Sample a fixed number of responses from the SFT baseline model.
2. Score each response with `compute_score`.
4. Convert the score into an implied problem rating relative to the anchor Elo of 1340.
5. Average or weighted-average the implied ratings across initializer models.

Suggested conversion:

```python
score = clip(success_fraction(group_rewards), eps, 1 - eps)
problem_rating = model_anchor_elo - 400 * log10(score / (1 - score))
```

Reasoning:

- If a model solves a problem easily, `score` is high and the inferred problem Elo is lower than the model Elo.
- If a model fails or solves late, `score` is low and the inferred problem Elo is higher than the model Elo.

Anchor choices:

- SFT baseline anchor: `1340`.

Store bootstrap ratings so RLOO training does not need to recompute them every run.

## Sampling Distribution

Sample training prompts from a Gaussian over problem Elo ratings centered at the current agent Elo:

```text
target_rating = R_agent
weight_i = exp(-0.5 * ((R_problem_i - target_rating) / elo_sigma) ** 2)
```

Use a 5% uniform exploration mixture:

```text
p_i = 0.95 * normalized_gaussian_weight_i + 0.05 * uniform_i
```

Sample without replacement. Repeats across different RLOO steps are allowed.

## Elo-Scale Sigma

The normalized curriculum-time sigma from the proposal is `0.5`. Map one normalized unit to the canonical 400-point Elo interval, since a 400-point Elo difference corresponds to a 10:1 odds ratio.

```text
elo_sigma = 0.5 * 400 = 200
```

Decision:

- Use `elo_sigma = 200`.

This is also a practical width: it focuses sampling near the current agent level but still includes a meaningful range of nearby problems.

## CLI Args

Exact names are flexible, but include at least:

- `--curriculum_type`, default `elo_gaussian` in `extension_trainer`.
- `--elo_initial_agent_rating`, default `1340`.
- `--elo_k_agent`, default `32`.
- `--elo_k_problem_base`, default `800`.
- `--elo_sigma`, default `200`.
- `--elo_uniform_mix`, default `0.05`.
- `--elo_bootstrap_path`, optional path to saved initial problem ratings.
- `--elo_state_path`, optional path for resuming curriculum state.

It is fine for `extension_trainer` to always use the curriculum path, but keeping `--curriculum_type uniform` is useful for debugging.

## Required Logging

Keep all existing RLOO logging and add:

- `curriculum/agent_elo`
- `curriculum/sampled_rating_mean`
- `curriculum/sampled_rating_std`
- `curriculum/batch_success_rate`
- `curriculum/batch_success_by_step_mean`
- `curriculum/problem_elo_mean`
- `curriculum/problem_elo_std`
- `curriculum/mean_problem_attempts`

Nice-to-have logging:

- `curriculum/sampled_rating_min`
- `curriculum/sampled_rating_max`
- `curriculum/target_rating`
- histogram of all problem ratings
- histogram of sampled problem ratings
