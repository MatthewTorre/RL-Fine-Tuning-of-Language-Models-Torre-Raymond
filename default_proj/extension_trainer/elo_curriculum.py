"""Elo-based curriculum sampler for Countdown RLOO.

The sampler keeps one moving agent rating plus one rating per training problem.
It samples prompts near the current agent rating using a Gaussian distribution,
with a small uniform mixture to keep exploration alive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def success_fraction(group_rewards: list[float] | np.ndarray) -> float:
    """Fraction of sampled responses that exactly solve the prompt."""
    rewards = np.asarray(group_rewards, dtype=np.float32)
    if rewards.size == 0:
        return 0.0
    return float(np.mean(rewards == 1.0))


@dataclass
class EloCurriculumConfig:
    """Hyperparameters controlling Elo updates and curriculum sampling."""

    initial_agent_rating: float = 1340.0
    initial_problem_rating: float = 1500.0
    k_agent: float = 32.0
    k_problem_base: float = 800.0
    sigma: float = 200.0
    uniform_mix: float = 0.05
    seed: int = 0


class EloCurriculumSampler:
    """Sample and update Countdown problem ratings with an Elo curriculum."""

    def __init__(
        self,
        num_problems: int,
        config: EloCurriculumConfig | None = None,
        initial_problem_ratings: np.ndarray | None = None,
    ):
        if num_problems <= 0:
            raise ValueError(f"num_problems must be positive, got {num_problems}")

        self.config = config or EloCurriculumConfig()
        self.num_problems = int(num_problems)
        self.agent_rating = float(self.config.initial_agent_rating)
        self.rng = np.random.default_rng(self.config.seed)

        if initial_problem_ratings is None:
            self.problem_ratings = np.full(
                self.num_problems,
                float(self.config.initial_problem_rating),
                dtype=np.float64,
            )
        else:
            self.problem_ratings = np.asarray(initial_problem_ratings, dtype=np.float64)
            if self.problem_ratings.shape != (self.num_problems,):
                raise ValueError(
                    "initial_problem_ratings must have shape "
                    f"({self.num_problems},), got {self.problem_ratings.shape}"
                )

        self.problem_attempts = np.zeros(self.num_problems, dtype=np.int64)
        self.problem_successes = np.zeros(self.num_problems, dtype=np.float64)
        self.last_probs = np.full(self.num_problems, 1.0 / self.num_problems, dtype=np.float64)

    @staticmethod
    def expected_score(agent_rating: float, problem_rating: float | np.ndarray) -> float | np.ndarray:
        """Expected agent score under the standard Elo logistic model."""
        return 1.0 / (1.0 + np.power(10.0, (problem_rating - agent_rating) / 400.0))

    def sampling_probs(self) -> np.ndarray:
        """Return Gaussian curriculum probabilities mixed with uniform exploration."""
        if self.config.sigma <= 0:
            raise ValueError(f"elo sigma must be positive, got {self.config.sigma}")
        if not 0.0 <= self.config.uniform_mix <= 1.0:
            raise ValueError(f"uniform_mix must be in [0, 1], got {self.config.uniform_mix}")

        target_rating = self.agent_rating
        z = (self.problem_ratings - target_rating) / self.config.sigma
        weights = np.exp(-0.5 * np.square(z))
        weight_sum = float(weights.sum())

        if not np.isfinite(weight_sum) or weight_sum <= 0:
            gaussian_probs = np.full(self.num_problems, 1.0 / self.num_problems, dtype=np.float64)
        else:
            gaussian_probs = weights / weight_sum

        uniform_probs = np.full(self.num_problems, 1.0 / self.num_problems, dtype=np.float64)
        probs = (1.0 - self.config.uniform_mix) * gaussian_probs + self.config.uniform_mix * uniform_probs
        probs = probs / probs.sum()
        self.last_probs = probs
        return probs

    def sample_ids(self, batch_size: int) -> np.ndarray:
        """Sample problem IDs with replacement using current curriculum probabilities."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        probs = self.sampling_probs()
        return self.rng.choice(self.num_problems, size=batch_size, replace=False, p=probs)

    def update(self, problem_ids: list[int] | np.ndarray, rewards_by_problem: list[list[float]]) -> dict[str, float]:
        """Update agent/problem Elo from a batch of grouped rewards."""
        if len(problem_ids) != len(rewards_by_problem):
            raise ValueError(
                f"problem_ids and rewards_by_problem length mismatch: "
                f"{len(problem_ids)} vs {len(rewards_by_problem)}"
            )

        if len(problem_ids) == 0:
            return self.metrics(np.asarray([], dtype=np.int64), [])

        agent_deltas = []
        success_scores = []
        problem_ids_array = np.asarray(problem_ids, dtype=np.int64)

        for problem_id, group_rewards in zip(problem_ids_array, rewards_by_problem):
            score = success_fraction(group_rewards)
            expected = float(self.expected_score(self.agent_rating, self.problem_ratings[problem_id]))

            self.problem_attempts[problem_id] += 1
            self.problem_successes[problem_id] += score

            k_problem = self.config.k_problem_base / float(self.problem_attempts[problem_id])
            rating_error = score - expected

            agent_deltas.append(self.config.k_agent * rating_error)
            self.problem_ratings[problem_id] -= k_problem * rating_error
            success_scores.append(score)

        # Average over the prompt batch so the agent K-factor is not implicitly
        # multiplied by batch size.
        self.agent_rating += float(np.mean(agent_deltas))
        return self.metrics(problem_ids_array, success_scores)

    def metrics(
        self,
        sampled_ids: np.ndarray,
        success_scores: list[float] | np.ndarray,
    ) -> dict[str, float]:
        """Build scalar metrics for W&B logging."""
        success_scores_array = np.asarray(success_scores, dtype=np.float64)
        if sampled_ids.size > 0:
            sampled_ratings = self.problem_ratings[sampled_ids]
        else:
            sampled_ratings = np.asarray([], dtype=np.float64)

        return {
            "agent_elo": float(self.agent_rating),
            "target_rating": float(self.agent_rating),
            "sampled_rating_mean": float(sampled_ratings.mean()) if sampled_ratings.size else 0.0,
            "sampled_rating_std": float(sampled_ratings.std()) if sampled_ratings.size else 0.0,
            "sampled_rating_min": float(sampled_ratings.min()) if sampled_ratings.size else 0.0,
            "sampled_rating_max": float(sampled_ratings.max()) if sampled_ratings.size else 0.0,
            "batch_success_rate": float(np.mean(success_scores_array > 0.0)) if success_scores_array.size else 0.0,
            "batch_success_fraction_mean": float(success_scores_array.mean()) if success_scores_array.size else 0.0,
            "problem_elo_mean": float(self.problem_ratings.mean()),
            "problem_elo_std": float(self.problem_ratings.std()),
            "mean_problem_attempts": float(self.problem_attempts.mean()),
            "unique_problem_fraction": float(np.mean(self.problem_attempts > 0)),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return JSON/NPZ-serializable curriculum state."""
        return {
            "agent_rating": self.agent_rating,
            "problem_ratings": self.problem_ratings,
            "problem_attempts": self.problem_attempts,
            "problem_successes": self.problem_successes,
            "last_probs": self.last_probs,
            "rng_state": self.rng.bit_generator.state,
            "config": self.config.__dict__.copy(),
        }

    def save(self, path: str | Path) -> None:
        """Save curriculum state to a compressed NumPy archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        np.savez_compressed(
            path,
            agent_rating=np.asarray(state["agent_rating"], dtype=np.float64),
            problem_ratings=state["problem_ratings"],
            problem_attempts=state["problem_attempts"],
            problem_successes=state["problem_successes"],
            last_probs=state["last_probs"],
            rng_state=np.asarray([state["rng_state"]], dtype=object),
            config=np.asarray([state["config"]], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EloCurriculumSampler":
        """Load curriculum state saved by `save`."""
        with np.load(path, allow_pickle=True) as data:
            config_dict = data["config"][0].item() if hasattr(data["config"][0], "item") else data["config"][0]
            sampler = cls(
                num_problems=int(data["problem_ratings"].shape[0]),
                config=EloCurriculumConfig(**config_dict),
                initial_problem_ratings=data["problem_ratings"],
            )
            sampler.agent_rating = float(data["agent_rating"])
            sampler.problem_attempts = data["problem_attempts"].astype(np.int64)
            sampler.problem_successes = data["problem_successes"].astype(np.float64)
            sampler.last_probs = data["last_probs"].astype(np.float64)
            rng_state = data["rng_state"][0].item() if hasattr(data["rng_state"][0], "item") else data["rng_state"][0]
            sampler.rng.bit_generator.state = rng_state
            return sampler


def load_problem_ratings(path: str | Path | None, num_problems: int) -> np.ndarray | None:
    """Load optional bootstrap problem ratings from `.npy`, `.npz`, or text."""
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Elo bootstrap path does not exist: {path}")

    if path.suffix == ".npy":
        ratings = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "problem_ratings" in data:
                ratings = data["problem_ratings"]
            else:
                first_key = next(iter(data.keys()))
                ratings = data[first_key]
    else:
        ratings = np.loadtxt(path)

    ratings = np.asarray(ratings, dtype=np.float64)
    if ratings.shape != (num_problems,):
        raise ValueError(f"Expected {num_problems} bootstrap ratings, got shape {ratings.shape}")
    return ratings
