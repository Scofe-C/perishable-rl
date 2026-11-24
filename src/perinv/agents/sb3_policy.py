"""A trained SB3 model as a Policy (SPEC §8): returns order quantities, never the raw action.

    policy = SB3Policy(model, "ppo", cfg, vecnormalize=frozen_or_None)

Guards against the team draft §7 pitfalls:
- VecNormalize is frozen: a private copy with training=False and norm_reward=False, so evaluation never updates
  the statistics and results cannot depend on path order.
- Actions are deterministic (model.predict(..., deterministic=True)).
- MaskablePPO gets action_masks from the raw state (k ≤ R), as at training time; DQN has no masks and relies on
  the env's clipping (SPEC §4.2).
- Box actions go through perinv.env.wrappers.capacity_share_orders, the same function the training wrapper uses.

Owner: Member 2.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np

from perinv.agents.factory import DISCRETE
from perinv.baselines.base import remaining_capacity
from perinv.env.config import EnvConfig
from perinv.env.wrappers import capacity_share_orders


def freeze(vecnormalize):
    """A copy of the VecNormalize statistics that can no longer change, detached from its training envs."""
    if vecnormalize is None:
        return None
    frozen = copy.copy(vecnormalize)
    frozen.obs_rms = copy.deepcopy(vecnormalize.obs_rms)
    frozen.ret_rms = copy.deepcopy(vecnormalize.ret_rms)
    frozen.training, frozen.norm_reward = False, False
    frozen.venv = None
    return frozen


def load_vecnormalize(path: str | Path):
    """VecNormalize.load needs a VecEnv; evaluation only needs the statistics, so unpickle and freeze."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return freeze(pickle.load(fh))


class SB3Policy:
    def __init__(self, model, algo: str, cfg: EnvConfig, vecnormalize=None):
        self.model, self.algo, self.cfg = model, algo, cfg
        self.vecnormalize = freeze(vecnormalize)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if self.vecnormalize is not None and self.vecnormalize.norm_obs:
            x = self.vecnormalize.normalize_obs(x).astype(np.float32)
        remaining = remaining_capacity(info, self.cfg.capacity)
        if self.algo == "maskppo":
            mask = np.arange(self.cfg.capacity + 1) <= remaining
            action, _ = self.model.predict(x, deterministic=True, action_masks=mask)
        else:
            action, _ = self.model.predict(x, deterministic=True)
        if self.algo in DISCRETE:
            return np.array([int(np.asarray(action).reshape(-1)[0])], dtype=np.int64)
        return capacity_share_orders(action, remaining)
