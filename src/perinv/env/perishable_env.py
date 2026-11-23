"""PerishableInventoryEnv: the Gymnasium env every agent trains and is evaluated on (SPEC §2–§6).

    env = PerishableInventoryEnv(cfg)                        # the runner's env factory: make_env(cfg)
    obs, info = env.reset(seed=k, options={"demand_path": demand[k]})   # evaluation: replay CRN path k
    obs, info = env.reset(seed=s)                           # training: Poisson demand from env.np_random (D5)
    obs, r, terminated, truncated, info = env.step(q)       # q: int[N] orders; terminated at t = T, never truncated

The day itself is perinv.env.dynamics.step; this class only keeps the state, draws or replays demand and
builds observations and the info dict. Wrappers for agents: perinv.env.wrappers.

Owner: Member 1. Contract: docs/SPEC.md.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from perinv.env import dynamics
from perinv.env.config import EnvConfig


class PerishableInventoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.n, self.L, self.P, self.T, self.C = cfg.n_skus, cfg.shelf_life, cfg.lead_time, cfg.horizon, cfg.capacity
        self.action_space = gym.spaces.MultiDiscrete([self.C + 1] * self.n)     # D6
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.n * (self.L + self.P) + 7 + 2,), np.float32)
        self._lam = cfg.daily_lambda()
        self._start = dynamics.initial_state(cfg)    # D4: warm start, pipeline pre-filled
        self.inv = np.zeros((self.n, self.L), np.int64)
        self.pipe = np.zeros((self.n, self.P), np.int64)
        self.t = 0
        self._demand: np.ndarray | None = None

    # ------------------------------------------------------------ state

    @property
    def occupancy(self) -> int:
        """Decision-time occupancy: on hand + in transit (SPEC §2)."""
        return int(self.inv.sum() + self.pipe.sum())

    @property
    def remaining(self) -> int:
        """R = C − occupancy: the most that can be ordered today, summed over SKUs."""
        return self.C - self.occupancy

    def _state(self) -> dict:
        return {"inv": self.inv.copy(), "pipe": self.pipe.copy(), "t": self.t}

    def _obs(self) -> np.ndarray:
        dow = np.zeros(7)
        dow[self.t % 7] = 1.0
        return np.concatenate([self.inv.ravel() / self.C, self.pipe.ravel() / self.C, dow,
                               [self.t / self.T, self.occupancy / self.C]]).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """bool[C+1]: order size k is allowed iff k ≤ R (SPEC §4.2, N = 1). Lives on the env for SubprocVecEnv."""
        if self.n != 1:
            raise ValueError("action_masks() is defined for N = 1 only (SPEC §4.2)")
        return np.arange(self.C + 1) <= self.remaining

    # ------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        path = (options or {}).get("demand_path")
        if path is None:   # D5: training mode, demand drawn now for the whole episode from the seeded generator
            self._demand = self.np_random.poisson(self._lam).astype(np.int64)
        else:
            path = np.asarray(path)
            if path.shape != (self.T, self.n):
                raise ValueError(f"demand_path must have shape (T, N) = {(self.T, self.n)}, got {path.shape}")
            if not np.issubdtype(path.dtype, np.integer) or (path < 0).any():
                raise ValueError("demand_path must hold non-negative integers")
            self._demand = path.astype(np.int64, copy=True)
        self.inv, self.pipe = (a.copy() for a in self._start)
        self.t = 0
        return self._obs(), {"state": self._state()}

    def step(self, action):
        if self._demand is None or self.t >= self.T:
            raise RuntimeError("call reset() before step() (the episode is over or never started)")
        q = np.asarray(action)
        if q.size != self.n or not np.issubdtype(q.dtype, np.integer):
            raise ValueError(f"action must be {self.n} integer order(s), got {q!r}")
        q = q.reshape(1, self.n).astype(np.int64)
        occ = self.occupancy
        d = self._demand[self.t]
        tr = dynamics.step(self.inv[None], self.pipe[None], q, d[None], self.cfg)
        self.inv, self.pipe = tr.inv[0], tr.pipe[0]
        self.t += 1
        info = {
            "sold": tr.sold[0], "demand": d.copy(), "lost": tr.lost[0], "expired": tr.expired[0],
            "ordered": tr.ordered[0], "held": tr.held[0],
            "occupancy": occ,                                   # D3: before today's order
            "clipped": bool(tr.clipped[0]),
            "reward_breakdown": {k: float(v[0]) for k, v in tr.breakdown.items()},
            "state": self._state(),
        }
        # D8: day T is a true end (no salvage, V[T] = 0 in the DP), so `terminated`. `truncated` would make SB3
        # bootstrap gamma * V(final state) and value stock that can never be sold.
        return self._obs(), float(tr.reward[0]), self.t == self.T, False, info
