"""Grouped bootstrap CIs and pairwise comparisons (spec §12).

Primary analysis unit is `intent_group_id`, not the individual query row
(canonical/semi_formal/naive_layperson are three paraphrases of one
observation). The bootstrap resamples *groups* with replacement and each
replicate's statistic is computed over every row belonging to the sampled
groups (a group sampled twice counts its rows twice) — never resamples
individual rows directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def mean_statistic(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    seed: int
    n_groups: int


def grouped_bootstrap(
    rows_by_group: dict[str, list[float]],
    statistic: Callable[[list[float]], float] = mean_statistic,
    n_resamples: int = 2000,
    seed: int = 20260817,
    ci_level: float = 0.95,
) -> BootstrapResult:
    group_ids = list(rows_by_group.keys())
    n_groups = len(group_ids)
    all_values = [v for g in group_ids for v in rows_by_group[g]]
    point_estimate = statistic(all_values)

    if n_groups == 0:
        return BootstrapResult(point_estimate=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                                n_resamples=n_resamples, seed=seed, n_groups=0)

    rng = _rng(seed)
    replicate_stats = np.empty(n_resamples)
    idx_array = np.arange(n_groups)
    for b in range(n_resamples):
        sampled_idx = rng.choice(idx_array, size=n_groups, replace=True)
        values: list[float] = []
        for i in sampled_idx:
            values.extend(rows_by_group[group_ids[i]])
        replicate_stats[b] = statistic(values)

    alpha = 1 - ci_level
    lo, hi = np.percentile(replicate_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point_estimate=point_estimate, ci_low=float(lo), ci_high=float(hi),
        n_resamples=n_resamples, seed=seed, n_groups=n_groups,
    )


@dataclass
class PairedComparisonResult:
    point_diff: float                # statistic(A) - statistic(B) on full data
    ci_low: float
    ci_high: float
    bootstrap_p_value: float
    n_resamples: int
    seed: int
    n_groups: int


def paired_grouped_bootstrap_diff(
    rows_by_group_a: dict[str, list[float]],
    rows_by_group_b: dict[str, list[float]],
    statistic: Callable[[list[float]], float] = mean_statistic,
    n_resamples: int = 2000,
    seed: int = 20260817,
    ci_level: float = 0.95,
) -> PairedComparisonResult:
    """Both dicts must share the same group_id keys (the intersection of
    intent groups both models/settings were actually scored on) so the same
    resampled group indices apply to both sides of the paired difference."""
    group_ids = sorted(set(rows_by_group_a.keys()) & set(rows_by_group_b.keys()))
    n_groups = len(group_ids)

    all_a = [v for g in group_ids for v in rows_by_group_a[g]]
    all_b = [v for g in group_ids for v in rows_by_group_b[g]]
    point_diff = statistic(all_a) - statistic(all_b)

    if n_groups == 0:
        return PairedComparisonResult(point_diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                                       bootstrap_p_value=float("nan"), n_resamples=n_resamples, seed=seed, n_groups=0)

    rng = _rng(seed)
    diffs = np.empty(n_resamples)
    idx_array = np.arange(n_groups)
    for b in range(n_resamples):
        sampled_idx = rng.choice(idx_array, size=n_groups, replace=True)
        values_a: list[float] = []
        values_b: list[float] = []
        for i in sampled_idx:
            g = group_ids[i]
            values_a.extend(rows_by_group_a[g])
            values_b.extend(rows_by_group_b[g])
        diffs[b] = statistic(values_a) - statistic(values_b)

    alpha = 1 - ci_level
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided bootstrap p-value: proportion of replicates on the other
    # side of zero from the point estimate, doubled, capped at 1.
    if point_diff >= 0:
        p_one_sided = float(np.mean(diffs <= 0))
    else:
        p_one_sided = float(np.mean(diffs >= 0))
    p_value = min(1.0, 2 * p_one_sided)

    return PairedComparisonResult(
        point_diff=float(point_diff), ci_low=float(lo), ci_high=float(hi),
        bootstrap_p_value=p_value, n_resamples=n_resamples, seed=seed, n_groups=n_groups,
    )


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict[str, float | bool]]:
    """Standard BH step-up procedure. Returns, in the *original* input
    order, {p_value, rank, adjusted_p_value, reject}."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    prev_min = 1.0
    for rank_from_largest, orig_idx in enumerate(reversed(indexed)):
        rank = m - rank_from_largest  # 1-indexed rank from smallest
        raw_adj = p_values[orig_idx] * m / rank
        prev_min = min(prev_min, raw_adj)
        adjusted[orig_idx] = prev_min

    out = []
    for i in range(m):
        out.append({
            "p_value": p_values[i],
            "adjusted_p_value": min(1.0, adjusted[i]),
            "reject_null": adjusted[i] <= alpha,
        })
    return out
