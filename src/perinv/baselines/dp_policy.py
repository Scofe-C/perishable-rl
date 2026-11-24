"""The exact DP as a Policy (N = 1): looks up pi[t, state] from perinv.dp.backward_induction.

    policy = dp_policy.build(ctx)    # runner builder (WAREHOUSE §7); solves ctx.cfg, never the eval world
    sol = dp_policy.load_or_solve(cfg)

Solutions are cached in results/dp/<config_hash>.npz and checked against the config hash and
EVENT_ORDER_VERSION on load; an unreadable or mismatched file counts as a miss and is re-solved. Several jobs may
build the DP for one config at once (e.g. shocks.yaml groups A_*): each writer uses its own temp file and then
renames it, and the solve is deterministic, so the last writer wins with identical content.
Override the folder with PERINV_DP_CACHE.

Owner: Member 1. Contract: docs/SPEC.md §8.
"""

from __future__ import annotations

import os
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

from perinv.dp import state_space
from perinv.dp.backward_induction import Solution, solve
from perinv.env.config import EnvConfig
from perinv.env.dynamics import EVENT_ORDER_VERSION
from perinv.utils.atomic import retry_on_permission_error

DEFAULT_CACHE = Path("results/dp")


def cache_dir() -> Path:
    return Path(os.environ.get("PERINV_DP_CACHE", DEFAULT_CACHE))


def cache_path(cfg: EnvConfig, root: Path | None = None) -> Path:
    return (root or cache_dir()) / f"{cfg.config_hash()}.npz"


def save(sol: Solution, cfg: EnvConfig, path: Path) -> None:
    """Write via a temp file unique to this writer (not utils.atomic's shared `<name>.tmp`: two jobs solving the
    same config would otherwise write and rename one temp file), then rename over the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as fh:   # file handle: np.savez would append .npz to the temp name
            np.savez_compressed(fh, V=sol.V, pi=sol.pi.astype(np.int16), config_hash=cfg.config_hash(),
                                event_order_version=EVENT_ORDER_VERSION)
        retry_on_permission_error(lambda: os.replace(tmp, path))   # Windows: a reader may hold the target
    finally:
        tmp.unlink(missing_ok=True)


def _load(path: Path, cfg: EnvConfig, n_states: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(V, pi) if the cache file is readable and belongs to this config; otherwise None (a miss)."""
    try:
        with np.load(path) as z:
            if (str(z["config_hash"]) == cfg.config_hash() and str(z["event_order_version"]) == EVENT_ORDER_VERSION
                    and z["V"].shape == (cfg.horizon + 1, n_states)):
                return z["V"], z["pi"].astype(np.int64)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    return None


def load_or_solve(cfg: EnvConfig, root: Path | None = None) -> tuple[Solution, dict]:
    """(solution, meta). meta: cache path, whether it was a cache hit, solve seconds."""
    path = cache_path(cfg, root)
    space = state_space.build(cfg.capacity, cfg.shelf_life, cfg.lead_time)
    cached = _load(path, cfg, len(space)) if path.exists() else None
    if cached is not None:
        return Solution(space, *cached), {"cache": str(path), "hit": True}
    t0 = time.perf_counter()
    sol = solve(cfg)
    seconds = time.perf_counter() - t0
    save(sol, cfg, path)
    return sol, {"cache": str(path), "hit": False, "solve_seconds": round(seconds, 3)}


class DPPolicy:
    """Optimal orders from the DP table. Reads the raw state from info["state"] (SPEC §5)."""

    def __init__(self, sol: Solution):
        self.sol = sol

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        s = info["state"]
        x = self.sol.space.from_env(np.asarray(s["inv"])[0], np.asarray(s["pipe"])[0])
        return np.array([self.sol.pi[s["t"], int(self.sol.space.index(x))]], dtype=np.int64)


def build(ctx) -> DPPolicy:
    """Runner builder. Solves the build world (ctx.cfg); in a cross-evaluation that is the policy's belief."""
    if ctx.cfg.n_skus != 1:
        raise ValueError(f"the DP policy is for N = 1 only, got N = {ctx.cfg.n_skus}")
    sol, meta = load_or_solve(ctx.cfg)
    ctx.run.update(dp=meta | {"value_initial_state": sol.initial_value(ctx.cfg)})
    ctx.run.log("dp_ready", **meta)
    return DPPolicy(sol)
