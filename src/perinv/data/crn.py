"""Common-random-number demand paths (docs/SPEC.md §7).

Every policy is evaluated on the same pre-generated paths, so profit differences can be paired per path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from perinv.data.scenarios import SHOCK_STREAM, apply_shocks
from perinv.env.config import EnvConfig

FORMAT_VERSION = 1
SPLIT_ID = {"tune": 0, "test": 1}
DEFAULT_SIZES = {"tune": 50, "test": 100}


def generate_paths(cfg: EnvConfig, split: str, n_paths: int, root_seed: int) -> np.ndarray:
    """int32[n_paths, T, N]. Path k depends only on (root_seed, split, k), never on n_paths.

    With a scenario, path k is the base path k plus the shocks, drawn from a separate rng stream
    (perinv.data.scenarios), so it stays paired with the unshocked path k.
    """
    lam = np.asarray(cfg.demand_mean)[None, :] * np.asarray(cfg.season)[np.arange(cfg.horizon) % 7, None]
    out = np.empty((n_paths, cfg.horizon, cfg.n_skus), dtype=np.int32)
    for k in range(n_paths):
        rng = np.random.default_rng([root_seed, SPLIT_ID[split], k])
        out[k] = rng.poisson(lam)
        if cfg.scenario:
            shock_rng = np.random.default_rng([root_seed, SPLIT_ID[split], k, SHOCK_STREAM])
            out[k] = apply_shocks(out[k], lam, cfg.scenario, shock_rng)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_paths(cfg: EnvConfig, root: str | Path = "data/crn", root_seed: int = 2026,
                sizes: dict[str, int] | None = None) -> Path:
    sizes = sizes or DEFAULT_SIZES
    out_dir = Path(root) / cfg.demand_hash()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, n in sizes.items():
        f = out_dir / f"{split}.npz"
        np.savez_compressed(f, demand=generate_paths(cfg, split, n, root_seed),
                            path_seed=np.arange(n, dtype=np.int64))
        files[split] = {"n_paths": n, "sha256": _sha256(f)}
    manifest = {"version": FORMAT_VERSION, "root_seed": root_seed,
                "params": cfg.demand_params(), "files": files}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out_dir


def load_paths(cfg: EnvConfig, split: str, root: str | Path = "data/crn") -> np.ndarray:
    """Load and verify against the manifest; refuses paths generated for other demand params."""
    out_dir = Path(root) / cfg.demand_hash()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if manifest["params"] != cfg.demand_params():
        raise ValueError(f"CRN params mismatch in {out_dir}")
    f = out_dir / f"{split}.npz"
    if _sha256(f) != manifest["files"][split]["sha256"]:
        raise ValueError(f"{f} does not match manifest hash; regenerate or restore it")
    return np.load(f)["demand"]
