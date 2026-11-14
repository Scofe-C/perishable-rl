"""Generate CRN demand paths for env configs: data/crn/<demand_hash>/{tune,test}.npz + manifest.json.

    python scripts/make_crn_paths.py configs/env/*.yaml
"""

import argparse

import yaml

from perinv.data.crn import write_paths
from perinv.env.config import load_env_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("env_configs", nargs="+")
    ap.add_argument("--budget", default="configs/budget.yaml")
    ap.add_argument("--root", default="data/crn")
    args = ap.parse_args()

    budget = yaml.safe_load(open(args.budget))
    seen = set()
    for path in args.env_configs:
        cfg = load_env_config(path)
        if cfg.demand_hash() in seen:  # several capacities share one demand process
            continue
        seen.add(cfg.demand_hash())
        out = write_paths(cfg, args.root, budget["crn_root_seed"], budget["crn_paths"])
        print(f"{path} -> {out}")


if __name__ == "__main__":
    main()
