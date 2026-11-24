"""Solve the exact DP for N = 1 env configs and cache it: results/dp/<config_hash>.npz.

    python scripts/solve_dp.py configs/env/n1_k06.yaml configs/env/n1_k10.yaml

The runner's DP builder uses the same cache, so this is optional (useful to inspect V and timings).

Owner: Member 1. Contract: docs/SPEC.md.
"""

import argparse
import sys

from perinv.baselines.dp_policy import load_or_solve
from perinv.env.config import load_env_config
from perinv.utils.seeding import init_process


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env_configs", nargs="+")
    args = ap.parse_args()
    for path in args.env_configs:
        cfg = load_env_config(path)
        if cfg.n_skus != 1:
            print(f"skip {path}: N = {cfg.n_skus} (the DP is N = 1 only)", file=sys.stderr)
            continue
        sol, meta = load_or_solve(cfg)
        how = "cached" if meta["hit"] else f"solved in {meta['solve_seconds']} s"
        print(f"{path}: C={cfg.capacity} states={len(sol.space)} V0={sol.initial_value(cfg):.2f} "
              f"({how}) -> {meta['cache']}")
    return 0


if __name__ == "__main__":
    init_process(0)
    sys.exit(main())
