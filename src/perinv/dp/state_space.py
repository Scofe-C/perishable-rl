"""N = 1 DP states (SPEC §2): the vector x = (inv[0 .. L-2], pipe[0 .. P-1]) with sum(x) ≤ C.

inv[L-1] is left out because it is always 0 at decision time, so there are C(C+D, D) states with D = L-1+P
(C=20, L=3, P=2: 10,626). States are sorted by their mixed-radix key, so lookups are a vectorized searchsorted.

Owner: Member 1. Contract: docs/SPEC.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class StateSpace:
    capacity: int
    shelf_life: int
    lead_time: int
    states: np.ndarray     # int64[S, D], sorted by key
    keys: np.ndarray       # int64[S], strictly increasing

    @property
    def dim(self) -> int:
        return self.shelf_life - 1 + self.lead_time

    def __len__(self) -> int:
        return len(self.states)

    def key(self, x: np.ndarray) -> np.ndarray:
        """Mixed-radix key of state vectors int[..., D] (radix C+1)."""
        x = np.asarray(x, dtype=np.int64)
        radix = (self.capacity + 1) ** np.arange(self.dim, dtype=np.int64)
        return x @ radix

    def index(self, x: np.ndarray) -> np.ndarray:
        """Row index of state vectors int[..., D]. Raises if any vector is not a state."""
        k = self.key(x)
        i = np.searchsorted(self.keys, k)
        bad = (i >= len(self.keys)) | (self.keys[np.minimum(i, len(self.keys) - 1)] != k)
        if np.any(bad):
            raise KeyError(f"not a DP state (negative, over capacity or inv[L-1] != 0): "
                           f"{np.asarray(x).reshape(-1, self.dim)[np.flatnonzero(np.ravel(bad))[0]]}")
        return i

    def from_env(self, inv: np.ndarray, pipe: np.ndarray) -> np.ndarray:
        """Env state (inv[..., L], pipe[..., P]) for one SKU -> DP vector. inv[..., L-1] must be 0."""
        inv, pipe = np.asarray(inv), np.asarray(pipe)
        if np.any(inv[..., -1] != 0):
            raise ValueError("inv[L-1] must be 0 at decision time (SPEC §2)")
        return np.concatenate([inv[..., :-1], pipe], axis=-1)

    def to_env(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """DP vectors int[B, D] -> env arrays inv int[B, 1, L], pipe int[B, 1, P]."""
        x = np.asarray(x, dtype=np.int64)
        b, L = len(x), self.shelf_life
        inv = np.zeros((b, 1, L), np.int64)
        inv[:, 0, :L - 1] = x[:, :L - 1]
        return inv, x[:, None, L - 1:].copy()

    def on_hand_after_arrival(self) -> np.ndarray:
        """int64[S]: stock available to sell today, sum(inv) + pipe[0] (SPEC §3 step 2)."""
        L = self.shelf_life
        return self.states[:, :L - 1].sum(axis=1) + self.states[:, L - 1]

    def occupancy(self) -> np.ndarray:
        return self.states.sum(axis=1)


def ranges(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(owner, offset): for each owner i, offsets 0 … lengths[i]-1, flattened."""
    owner = np.repeat(np.arange(len(lengths)), lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return owner, offset


def expected_count(capacity: int, shelf_life: int, lead_time: int) -> int:
    d = shelf_life - 1 + lead_time
    return comb(capacity + d, d)


def build(capacity: int, shelf_life: int, lead_time: int) -> StateSpace:
    """Every non-negative integer vector of length D = L-1+P with sum ≤ C, sorted by key."""
    d = shelf_life - 1 + lead_time
    states = np.zeros((1, 0), np.int64)
    for _ in range(d):   # extend by one coordinate 0 … C − (sum so far)
        owner, nxt = ranges(capacity - states.sum(axis=1) + 1)
        states = np.concatenate([states[owner], nxt[:, None]], axis=1)
    radix = (capacity + 1) ** np.arange(d, dtype=np.int64)
    keys = states @ radix
    order = np.argsort(keys)
    space = StateSpace(capacity, shelf_life, lead_time, states[order], keys[order])
    assert len(space) == expected_count(capacity, shelf_life, lead_time)
    return space
