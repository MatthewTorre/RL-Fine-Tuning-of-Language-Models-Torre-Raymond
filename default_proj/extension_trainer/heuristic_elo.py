"""Fast input-only problem Elo initialization for Countdown tasks."""

from __future__ import annotations

import ast
import math
from itertools import combinations
from typing import Any

import numpy as np


def _extract_problem_fields(ground_truth: Any) -> tuple[list[float], float]:
    """Return numbers and target from common Countdown metadata shapes."""
    if isinstance(ground_truth, str):
        ground_truth = ast.literal_eval(ground_truth)

    if not isinstance(ground_truth, dict):
        raise ValueError(f"Expected dict-like ground truth, got {type(ground_truth).__name__}")

    numbers = ground_truth.get("numbers", ground_truth.get("nums"))
    target = ground_truth.get("target")
    if numbers is None or target is None:
        raise ValueError(f"Ground truth must contain numbers/nums and target, got {ground_truth}")

    return [float(number) for number in numbers], float(target)


def _closeness(value: float, target: float) -> float:
    scale = max(10.0, 0.1 * abs(target), 1.0)
    return math.exp(-abs(value - target) / scale)


def _best_closeness(values: list[float], target: float) -> float:
    if not values:
        return 0.0
    return max(_closeness(value, target) for value in values)


def _gcd_affinity(numbers: list[float], target: float) -> float:
    target_int = int(round(abs(target)))
    if target_int == 0:
        return 0.0

    affinities = []
    for number in numbers:
        number_int = int(round(abs(number)))
        if number_int == 0:
            continue
        gcd_value = math.gcd(target_int, number_int)
        shared_factor = gcd_value / max(1, min(target_int, number_int))
        divides_bonus = 1.0 if target_int % number_int == 0 or number_int % target_int == 0 else 0.0
        affinities.append(0.7 * shared_factor + 0.3 * divides_bonus)

    return max(affinities) if affinities else 0.0


def heuristic_easy_score(ground_truth: Any) -> float:
    """Estimate problem easiness using only input numbers and the target."""
    numbers, target = _extract_problem_fields(ground_truth)
    abs_numbers = [abs(number) for number in numbers]

    pair_add_sub = []
    pair_mul_div = []
    for a, b in combinations(numbers, 2):
        pair_add_sub.extend([a + b, abs(a - b)])
        pair_mul_div.append(a * b)
        if b != 0:
            pair_mul_div.append(a / b)
        if a != 0:
            pair_mul_div.append(b / a)

    aggregates = [sum(numbers)]
    if abs_numbers:
        aggregates.extend([max(abs_numbers), float(np.prod(numbers))])

    single_closeness = _best_closeness(numbers, target)
    add_sub_closeness = _best_closeness(pair_add_sub, target)
    mul_div_closeness = _best_closeness(pair_mul_div, target)
    aggregate_closeness = _best_closeness(aggregates, target)
    gcd_score = _gcd_affinity(numbers, target)

    max_number = max(abs_numbers) if abs_numbers else 0.0
    magnitude_gap = abs(math.log1p(abs(target)) - math.log1p(max_number))
    mean_abs_number = float(np.mean(abs_numbers)) if abs_numbers else 0.0
    spread = float(np.std(abs_numbers) / max(mean_abs_number, 1.0)) if abs_numbers else 0.0

    return (
        1.5 * single_closeness
        + 1.0 * add_sub_closeness
        + 0.8 * mul_div_closeness
        + 0.5 * gcd_score
        + 0.3 * aggregate_closeness
        - 0.4 * magnitude_gap
        - 0.3 * spread
    )


def compute_heuristic_problem_ratings(
    dataset: Any,
    base_rating: float = 1500.0,
    scale: float = 200.0,
    min_rating: float = 1200.0,
    max_rating: float = 1800.0,
) -> np.ndarray:
    """Map cheap input-only easiness scores to initial problem Elo ratings."""
    if min_rating > max_rating:
        raise ValueError(f"min_rating must be <= max_rating, got {min_rating} > {max_rating}")
    if scale < 0:
        raise ValueError(f"scale must be non-negative, got {scale}")

    scores = np.asarray(
        [heuristic_easy_score(ground_truth) for ground_truth in dataset.all_ground_truth],
        dtype=np.float64,
    )
    score_std = float(scores.std())
    if score_std == 0.0:
        ratings = np.full(scores.shape, float(base_rating), dtype=np.float64)
    else:
        z_scores = (scores - float(scores.mean())) / score_std
        # Easier problems should start below the base problem rating.
        ratings = float(base_rating) - float(scale) * z_scores

    return np.clip(ratings, float(min_rating), float(max_rating)).astype(np.float64)
