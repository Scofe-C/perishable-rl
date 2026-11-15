"""The only evaluation loop (CLAUDE.md "one evaluation path"). Contract: docs/WAREHOUSE.md §3, SPEC §3/§6/§7/§8.

    result = evaluate(policy, env, demand)        # demand: int32[n_paths, T, N] from data.crn.load_paths
    result.per_path()                             # one row per path, for quick feedback
    writer.write_eval(result, cfg, run_key, path) # validated per-step fact file

Owner: Member 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from perinv.baselines.base import Policy
from perinv.utils.seeding import set_all_seeds

# Per-SKU quantities recorded every day, in the order they appear in fact_step (WAREHOUSE.md §3).
QTY_FIELDS = ("occ", "order", "sold", "lost", "expired", "held")
# fact column -> key in the env's step() info dict (SPEC §6)
_INFO_KEYS = {"order": "ordered", "sold": "sold", "lost": "lost", "expired": "expired", "held": "held"}


@dataclass
class EvalResult:
    """Raw per-step arrays. Index 0 follows `path_ids` order (not necessarily sorted)."""

    path_ids: np.ndarray            # int64[P]: CRN path index k
    qty: dict[str, np.ndarray]      # field -> int32[P, T, N]
    day_reward: np.ndarray          # float64[P, T]
    clipped: np.ndarray             # bool[P, T]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.qty["sold"].shape

    def per_path(self) -> pd.DataFrame:
        """One row per path, sorted by path_id. Profit is the sum of the env's own daily rewards."""
        sums = {f"total_{f}": self.qty[f].sum(axis=(1, 2)) for f in ("order", "sold", "lost", "expired")}
        df = pd.DataFrame({
            "path_id": self.path_ids,
            "profit": self.day_reward.sum(axis=1),
            **sums,
            "total_demand": sums["total_sold"] + sums["total_lost"],
            "clipped_days": self.clipped.sum(axis=1),
        })
        return df.sort_values("path_id", ignore_index=True)


def _per_sku(info: dict, key: str, n: int) -> np.ndarray:
    if key not in info:
        raise KeyError(f"env info is missing {key!r} (SPEC §6)")
    v = np.asarray(info[key])
    if v.size != n:
        raise ValueError(f"info[{key!r}] must have one value per SKU ({n}), got shape {v.shape}")
    return v.reshape(n)


def _occupancy(info: dict, n: int) -> np.ndarray:
    """Per-SKU occupancy at decision time, from the raw state: sum(inv[i]) + sum(pipe[i]) (SPEC §2)."""
    if "state" not in info:
        raise KeyError("env info is missing 'state' (SPEC §5: raw state always in info['state'])")
    s = info["state"]
    return (np.asarray(s["inv"]).reshape(n, -1).sum(axis=1)
            + np.asarray(s["pipe"]).reshape(n, -1).sum(axis=1))


def evaluate(policy: Policy, env, demand: np.ndarray, path_ids: Sequence[int] | None = None) -> EvalResult:
    """Run `policy` on each CRN path and record every day.

    `demand` is the full split array (int32[n_paths, T, N]). `path_ids` picks which paths to run, in that
    order; the default is all of them. Each path is seeded with its index k (SPEC §7), so results do not
    depend on the order or the subset of paths.
    """
    demand = np.asarray(demand)
    if demand.ndim != 3:
        raise ValueError(f"demand must be [n_paths, T, N], got shape {demand.shape}")
    ids = np.arange(demand.shape[0]) if path_ids is None else np.asarray(path_ids, dtype=np.int64)
    p_count, (_, horizon, n) = len(ids), demand.shape

    qty = {f: np.zeros((p_count, horizon, n), dtype=np.int32) for f in QTY_FIELDS}
    day_reward = np.zeros((p_count, horizon), dtype=np.float64)
    clipped = np.zeros((p_count, horizon), dtype=bool)

    for row, k in enumerate(ids):
        k = int(k)
        set_all_seeds(k)  # stochastic policies draw from the global RNGs (SPEC §7: seeded with k)
        policy.reset()
        obs, info = env.reset(seed=k, options={"demand_path": demand[k]})
        for t in range(horizon):
            qty["occ"][row, t] = _occupancy(info, n)
            action = policy.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            for field, key in _INFO_KEYS.items():
                qty[field][row, t] = _per_sku(info, key, n)
            day_reward[row, t] = float(reward)
            clipped[row, t] = bool(info.get("clipped", False))
            last = t == horizon - 1
            if terminated or truncated != last:
                raise RuntimeError(f"path {k}: episode ended at t={t} (terminated={terminated}, "
                                   f"truncated={truncated}); SPEC §3 truncates exactly at t=T")

    return EvalResult(path_ids=ids, qty=qty, day_reward=day_reward, clipped=clipped)
