"""Ensemble aggregation with calibration guardrails.

Binary:   trimmed mean in log-odds space (robust to one wild sample,
          and averaging in log-odds is less prone to the dilution toward
          0.5 that plain probability averaging causes).
MC:       trimmed mean of log-probabilities per option, renormalized,
          per-option floor applied.
Numeric:  pointwise trimmed mean of the sample CDFs (a convex combination
          of monotone CDFs is monotone).
"""

from __future__ import annotations

import math

import numpy as np

from . import config


def _trimmed_mean(values: list[float], trim_fraction: float) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    k = int(len(arr) * trim_fraction)
    if len(arr) - 2 * k < 1:
        k = 0
    return float(arr[k : len(arr) - k].mean())


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def aggregate_binary(probabilities: list[float]) -> float:
    """probabilities are decimals in (0,1). Returns clamped decimal."""
    log_odds = [_logit(p) for p in probabilities]
    p = _sigmoid(_trimmed_mean(log_odds, config.TRIM_FRACTION))
    return float(min(max(p, config.BINARY_FLOOR), config.BINARY_CEIL))


def aggregate_multiple_choice(
    samples: list[dict[str, float]], options: list[str]
) -> dict[str, float]:
    """samples: list of {option: prob} dicts (each ~sums to 1)."""
    agg: dict[str, float] = {}
    for opt in options:
        logs = [math.log(max(s.get(opt, 0.0), 1e-6)) for s in samples]
        agg[opt] = math.exp(_trimmed_mean(logs, config.TRIM_FRACTION))
    total = sum(agg.values())
    agg = {k: v / total for k, v in agg.items()}
    # Floor + renormalize (never zero-out an option), then fix fp drift.
    agg = {k: max(v, config.MC_FLOOR) for k, v in agg.items()}
    total = sum(agg.values())
    agg = {k: v / total for k, v in agg.items()}
    drift = 1.0 - sum(agg.values())
    last = options[-1]
    agg[last] += drift
    return agg


def aggregate_cdfs(cdfs: list[list[float]]) -> list[float]:
    arr = np.asarray(cdfs, dtype=float)  # (n_samples, cdf_size)
    n = arr.shape[0]
    k = int(n * config.TRIM_FRACTION)
    if n - 2 * k < 1:
        k = 0
    sorted_arr = np.sort(arr, axis=0)
    trimmed = sorted_arr[k : n - k, :].mean(axis=0)
    # Enforce monotone non-decreasing with the platform's min step, then re-round.
    out = np.maximum.accumulate(trimmed)
    return np.round(out, 10).tolist()
