"""The one interface every evaluated agent implements (docs/SPEC.md §8)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    def reset(self) -> None:
        """Called at the start of every evaluation path."""

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Return int64[N] orders. They must already be feasible: occupancy + sum <= C."""


def remaining_capacity(info: dict, capacity: int) -> int:
    """R = C − (on hand + in transit), from the raw state in info (SPEC §2, §5)."""
    s = info["state"]
    return int(capacity - np.asarray(s["inv"]).sum() - np.asarray(s["pipe"]).sum())


def inventory_position(info: dict) -> np.ndarray:
    """Per SKU: on hand + in transit (including today's arrival). int64[N]."""
    s = info["state"]
    n = np.asarray(s["inv"]).shape[0]
    return np.asarray(s["inv"]).reshape(n, -1).sum(axis=1) + np.asarray(s["pipe"]).reshape(n, -1).sum(axis=1)


def fit_to_capacity(q: np.ndarray, remaining: int) -> np.ndarray:
    """Heuristic orders scaled down to sum to at most R, keeping their proportions (largest remainder)."""
    from perinv.env.wrappers import largest_remainder

    q = np.maximum(np.asarray(q, dtype=np.int64), 0)
    remaining = max(int(remaining), 0)
    if q.sum() <= remaining:
        return q
    return largest_remainder(q / q.sum(), remaining)
