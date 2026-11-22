"""The one transition implementation (SPEC §3). The env and the DP both call `step`; nothing else may
re-implement the event order (CLAUDE.md hard rule; `tests/fakes.py` is held to it by a conformance test).

Every array has a leading batch axis B, so the env calls it with B = 1 and the DP builds its whole transition
table in one call.

    inv  int[B, N, L]   on hand by remaining shelf life (index 0 expires today)
    pipe int[B, N, P]   orders in transit, pipe[:, :, 0] arrives today (P = lead time)
    q    int[B, N]      orders chosen today (clipped here if they exceed capacity)
    d    int[B, N]      today's demand

Owner: Member 1. Contract: docs/SPEC.md §2, §3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perinv.env.config import EnvConfig

# Bump whenever SPEC §3 changes: it is part of every train_key (perinv.utils.run_io).
EVENT_ORDER_VERSION = "1"


@dataclass(frozen=True)
class Transition:
    """Result of one day for a batch. Next state plus the SPEC §6 quantities, all per row (and per SKU)."""

    inv: np.ndarray        # int64[B, N, L], next day's on hand (inv[:, :, L-1] == 0)
    pipe: np.ndarray       # int64[B, N, P]
    ordered: np.ndarray    # int64[B, N], after clipping
    sold: np.ndarray       # int64[B, N]
    lost: np.ndarray       # int64[B, N]
    expired: np.ndarray    # int64[B, N]
    held: np.ndarray       # int64[B, N], on hand after expiry
    clipped: np.ndarray    # bool[B]
    reward: np.ndarray     # float64[B] = revenue - ordering - holding - expiry - lost_sales
    breakdown: dict[str, np.ndarray]   # float64[B] each; revenue is income, the other four are costs (>= 0)


def clip_orders(q: np.ndarray, occ: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """SPEC §3 step 1 (D2): while sum(q) > R, take one unit off the largest order (ties: lowest SKU index).

    q int[B, N] >= 0, occ int[B] <= capacity. Returns (clipped orders int64[B, N], clipped bool[B]).
    """
    q = np.array(q, dtype=np.int64, copy=True)
    remaining = capacity - np.asarray(occ, dtype=np.int64)
    excess = q.sum(axis=1) - remaining
    clipped = excess > 0
    rows = np.flatnonzero(clipped)
    while rows.size:
        cols = np.argmax(q[rows], axis=1)      # argmax returns the first maximum: the lowest index on ties
        q[rows, cols] -= 1
        excess[rows] -= 1
        rows = rows[excess[rows] > 0]
    return q, clipped


def _check(inv: np.ndarray, pipe: np.ndarray, q: np.ndarray, d: np.ndarray, cfg: EnvConfig) -> None:
    b, n, L, P = inv.shape[0], cfg.n_skus, cfg.shelf_life, cfg.lead_time
    for name, a, shape in (("inv", inv, (b, n, L)), ("pipe", pipe, (b, n, P)), ("q", q, (b, n)), ("d", d, (b, n))):
        if a.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"{name} must be integer, got {a.dtype}")
        if (a < 0).any():
            raise ValueError(f"{name} must be >= 0")
    occ = inv.sum(axis=(1, 2)) + pipe.sum(axis=(1, 2))
    if (occ > cfg.capacity).any():
        raise ValueError(f"state occupancy {int(occ.max())} exceeds capacity {cfg.capacity}: not a reachable state")


def step(inv: np.ndarray, pipe: np.ndarray, q: np.ndarray, d: np.ndarray, cfg: EnvConfig) -> Transition:
    """One day for every row of the batch, in the SPEC §3 order. Inputs are not modified."""
    inv, pipe, q, d = (np.asarray(a) for a in (inv, pipe, q, d))
    _check(inv, pipe, q, d, cfg)
    inv = inv.astype(np.int64, copy=True)
    pipe = pipe.astype(np.int64, copy=True)

    # 1. decide: clip to the remaining capacity
    occ = inv.sum(axis=(1, 2)) + pipe.sum(axis=(1, 2))
    ordered, clipped = clip_orders(q, occ, cfg.capacity)
    # 2. arrive into the freshest bucket
    inv[:, :, -1] += pipe[:, :, 0]
    # 3. demand, issued FIFO from the oldest bucket
    left = d.astype(np.int64, copy=True)
    for k in range(cfg.shelf_life):
        take = np.minimum(inv[:, :, k], left)
        inv[:, :, k] -= take
        left -= take
    lost = left
    sold = d - lost
    # 4. expire the oldest bucket, then age the rest
    expired = inv[:, :, 0].copy()
    inv[:, :, :-1] = inv[:, :, 1:]
    inv[:, :, -1] = 0
    # 5. hold
    held = inv.sum(axis=2)
    # 6. pipeline: shift and add today's orders
    pipe[:, :, :-1] = pipe[:, :, 1:]
    pipe[:, :, -1] = ordered
    # 7. reward
    def cost(name: str, qty: np.ndarray) -> np.ndarray:
        return (np.asarray(getattr(cfg, name), dtype=np.float64) * qty).sum(axis=1)

    breakdown = {"revenue": cost("price", sold), "ordering": cost("order_cost", ordered),
                 "holding": cost("holding_cost", held), "expiry": cost("expiry_cost", expired),
                 "lost_sales": cost("lost_sales_cost", lost)}
    reward = (breakdown["revenue"] - breakdown["ordering"] - breakdown["holding"] - breakdown["expiry"]
              - breakdown["lost_sales"])
    return Transition(inv=inv, pipe=pipe, ordered=ordered, sold=sold.astype(np.int64), lost=lost,
                      expired=expired, held=held, clipped=clipped, reward=reward, breakdown=breakdown)
