"""IQM + bootstrap CIs over seeds; paired per-path differences (docs/EXPERIMENTS.md rule 5).

IQM = scipy.stats.trim_mean(x, 0.25); CI = scipy.stats.bootstrap. rliable is not used (DECISIONS.md, 2026-11-13).
Percentile bootstrap: with 3-5 seeds the BCa correction is unstable, and percentile is what rliable reports.

Owner: Member 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

N_RESAMPLES = 10_000
CONFIDENCE = 0.95


def iqm(x, axis: int = 0) -> np.ndarray:
    """Interquartile mean: drop the lowest and highest 25%, average the rest."""
    return sps.trim_mean(x, 0.25, axis=axis)


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float
    n: int


def bootstrap_ci(x, statistic=iqm, *, seed: int = 0, n_resamples: int = N_RESAMPLES,
                 confidence: float = CONFIDENCE) -> Estimate:
    """Point estimate + percentile bootstrap CI. Fixed seed, so reports are reproducible.

    With one value there is no spread to resample: the CI collapses to the value.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return Estimate(np.nan, np.nan, np.nan, 0)
    value = float(statistic(x))
    if x.size == 1 or np.all(x == x[0]):
        return Estimate(value, value, value, int(x.size))
    res = sps.bootstrap((x,), lambda s, axis: statistic(s, axis=axis), n_resamples=n_resamples,
                        confidence_level=confidence, method="percentile", rng=np.random.default_rng(seed))
    ci = res.confidence_interval
    return Estimate(value, float(ci.low), float(ci.high), int(x.size))


def paired_difference(a, b, *, seed: int = 0) -> Estimate:
    """Mean of per-path differences a - b (same CRN paths), with a bootstrap CI. a and b must be aligned."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align path by path, got {a.shape} vs {b.shape}")
    return bootstrap_ci(a - b, statistic=lambda d, axis=0: np.mean(d, axis=axis), seed=seed)
