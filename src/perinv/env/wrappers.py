"""Action wrappers that translate agent actions into the base env's integer orders (SPEC §4).

    CapacityShareAction(env)   Box(-1, 1, (N+1,)) → q[N]   PPO, A2C, SAC (one shared mapping, D1)
    DiscreteOrder(env)         Discrete(C+1) → q = [a]     DQN, MaskablePPO (N = 1); masks stay on the base env

`capacity_share_orders(a, R)` is the mapping itself, as a pure function. The SB3 policy adapter must use it
too, so that evaluate() receives order quantities (SPEC §8), never the raw Box vector.

Owner: Member 1. Contract: docs/SPEC.md §4.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

SHARE_SCALE = 4.0
_LO = 1.0 / (1.0 + np.exp(SHARE_SCALE))     # σ(−4)
_HI = 1.0 / (1.0 + np.exp(-SHARE_SCALE))    # σ(4)
# D7: softmax(4 · a[1:]). With a in [−1, 1], plain softmax would cap one SKU's weight at e/(e + (N−1)/e):
# 65 % for N = 5. Scale 4 lifts the cap to 99.9 %; a[1:] = 0 is still an even split.
SPLIT_SCALE = 4.0


def total_share(a0: float) -> float:
    """D1: rescaled sigmoid, exactly 0 at a0 = −1 and exactly 1 at a0 = +1 (a0 is clipped to [−1, 1])."""
    a0 = min(max(float(a0), -1.0), 1.0)
    return min(max(float((1.0 / (1.0 + np.exp(-SHARE_SCALE * a0)) - _LO) / (_HI - _LO)), 0.0), 1.0)


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integers ≥ 0 proportional to `weights` (sum 1) that add up to exactly `total`.
    Leftover units go to the largest fractional parts; ties go to the lowest index."""
    exact = weights * total
    q = np.floor(exact).astype(np.int64)
    short = total - int(q.sum())
    if short > 0:
        order = np.argsort(-(exact - q), kind="stable")    # stable: equal remainders keep index order
        q[order[:short]] += 1
    return q


def capacity_share_orders(action: np.ndarray, remaining: int) -> np.ndarray:
    """SPEC §4.1: a[0] sets the share of R to fill today; softmax(a[1:]) splits it over the SKUs."""
    a = np.asarray(action, dtype=np.float64).ravel()
    if a.size < 2 or not np.isfinite(a).all():
        raise ValueError(f"CapacityShareAction needs N+1 finite numbers, got {action!r}")
    remaining = int(remaining)
    total = int(np.floor(total_share(a[0]) * remaining + 0.5))    # share in [0, 1], so total in [0, R]
    z = SPLIT_SCALE * np.clip(a[1:], -1.0, 1.0)
    w = np.exp(z - z.max())
    return largest_remainder(w / w.sum(), total)


def _base(env: gym.Env):
    base = env.unwrapped
    if not hasattr(base, "remaining"):
        raise TypeError(f"{type(base).__name__} is not a PerishableInventoryEnv")
    return base


class CapacityShareAction(gym.ActionWrapper):
    """Box(-1, 1, (N+1,)) actions for PPO, A2C and SAC. Feasible by construction: sum(q) ≤ R."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._base = _base(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self._base.n + 1,), np.float32)

    def action(self, action):
        return capacity_share_orders(action, self._base.remaining)


class DiscreteOrder(gym.ActionWrapper):
    """Discrete(C+1) actions for DQN and MaskablePPO, N = 1 only. MaskablePPO finds the base env's
    action_masks() through get_wrapper_attr, also under Monitor and SubprocVecEnv."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = _base(env)
        if base.n != 1:
            raise ValueError("DiscreteOrder is for N = 1 only: (C+1)^N joint actions are infeasible (SPEC §4.2)")
        self.action_space = gym.spaces.Discrete(base.C + 1)

    def action(self, action):
        return np.array([int(action)], dtype=np.int64)
