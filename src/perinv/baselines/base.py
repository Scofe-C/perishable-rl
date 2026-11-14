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
