"""Measure O_ref (SPEC §1.1) and propose C = round(κ · O_ref) for env configs. Reports; never edits configs.

    python scripts/measure_oref.py configs/env/n*_k*.yaml configs/env/calibrated/*.yaml
        --json configs/env/oref_provisional.json      # (one line) read by make_calibrated_configs.py --oref

PROVISIONAL STAND-IN. SPEC defines O_ref with M2's unconstrained (s,S) policy, which does not exist yet. This
script uses a per-SKU order-up-to policy instead (order S_i − position_i, where position = on hand + in transit),
which is (s,S) with s = S − 1. Without a capacity limit the SKUs are independent, so each S_i is grid-searched
on its own on the `tune` paths (profit of that SKU alone). O_ref = mean over tune paths and days of the
post-order occupancy, sum(inv) + sum(pipe) + sum(q), summed over SKUs. Remeasure once M2's (s,S) exists.

Owner: Member 1 (provisional, see docs/STATUS.md M1 checklist).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from perinv.data.crn import load_paths
from perinv.env.config import EnvConfig, load_env_config
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.eval.evaluate import evaluate
from perinv.utils.seeding import init_process

UNCONSTRAINED = 10**6   # capacity that never binds


class OrderUpTo:
    """q_i = max(0, S_i − (on hand_i + in transit_i)). Deterministic; reads the raw state (SPEC §5)."""

    def __init__(self, levels):
        self.levels = np.asarray(levels, np.int64)

    def reset(self) -> None:
        pass

    def act(self, obs, info):
        s = info["state"]
        position = np.asarray(s["inv"]).sum(axis=1) + np.asarray(s["pipe"]).sum(axis=1)
        return np.maximum(self.levels - position, 0)


@dataclass(frozen=True)
class ORef:
    levels: list[int]           # best S_i per SKU
    o_ref: float                # mean post-order occupancy, summed over SKUs
    mean_profit: float          # of the stand-in at those levels, unconstrained


def unconstrained(cfg: EnvConfig) -> EnvConfig:
    return replace(cfg, capacity=UNCONSTRAINED, kappa=None, name=f"{cfg.name}-unconstrained")


def _single_sku(cfg: EnvConfig, i: int) -> EnvConfig:
    pick = lambda name: (getattr(cfg, name)[i],)   # noqa: E731
    return replace(cfg, n_skus=1, demand_mean=pick("demand_mean"), price=pick("price"), order_cost=pick("order_cost"),
                   holding_cost=pick("holding_cost"), expiry_cost=pick("expiry_cost"),
                   lost_sales_cost=pick("lost_sales_cost"), scenario=())


def best_level(cfg: EnvConfig, i: int, demand: np.ndarray, max_level: int | None = None) -> tuple[int, float]:
    """Grid-search S for SKU i alone (unconstrained, so the SKUs do not interact). Ties: the smaller S."""
    one = unconstrained(_single_sku(cfg, i))
    lam_max = cfg.demand_mean[i] * max(cfg.season)
    top = max_level if max_level is not None else int(np.ceil(lam_max * (cfg.lead_time + cfg.shelf_life) * 2)) + 5
    env, d = PerishableInventoryEnv(one), demand[:, :, i:i + 1]
    profits = [evaluate(OrderUpTo([s]), env, d).per_path()["profit"].mean() for s in range(top + 1)]
    best = int(np.argmax(profits))
    if best == top:
        raise RuntimeError(f"best S hit the grid edge {top} for SKU {i} of {cfg.name}: widen the grid")
    return best, float(profits[best])


def measure(cfg: EnvConfig, demand: np.ndarray) -> ORef:
    levels = [best_level(cfg, i, demand)[0] for i in range(cfg.n_skus)]
    res = evaluate(OrderUpTo(levels), PerishableInventoryEnv(unconstrained(cfg)), demand)
    post_order = (res.qty["occ"] + res.qty["order"]).sum(axis=2)        # [path, day]
    return ORef(levels, float(post_order.mean()), float(res.per_path()["profit"].mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env_configs", nargs="+")
    ap.add_argument("--crn-root", default="data/crn")
    ap.add_argument("--json", help="also write {config name: {demand_hash, levels, o_ref}} to this file")
    args = ap.parse_args()
    cache: dict[str, ORef] = {}
    out: dict[str, dict] = {}
    print(f"{'config':<22} {'N':>2} {'kappa':>5} {'C now':>6} {'S per SKU':<18} {'O_ref':>7} {'C proposed':>10}")
    for path in args.env_configs:
        cfg = load_env_config(path)
        if cfg.demand_hash() not in cache:        # O_ref depends on demand and costs, not on capacity
            cache[cfg.demand_hash()] = measure(cfg, load_paths(cfg, "tune", args.crn_root))
        r = cache[cfg.demand_hash()]
        proposed = max(1, round((cfg.kappa or 1.0) * r.o_ref))
        print(f"{cfg.name:<22} {cfg.n_skus:>2} {cfg.kappa!s:>5} {cfg.capacity:>6} {str(r.levels):<18} "
              f"{r.o_ref:>7.2f} {proposed:>10}")
        out[cfg.name] = {"demand_hash": cfg.demand_hash(), "n_skus": cfg.n_skus, "levels": r.levels,
                         "o_ref": r.o_ref}
    if args.json:
        doc = {"_note": "PROVISIONAL: order-up-to stand-in for M2's (s,S) (scripts/measure_oref.py). Remeasure.",
               "configs": out}
        Path(args.json).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    init_process(0)
    sys.exit(main())
