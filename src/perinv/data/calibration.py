"""Calibrated env configs from real-data demand parameters (contract C1: demand_params.parquet from infra/).

    python scripts/make_calibrated_configs.py <path>/demand_params.parquet

Rules (all deterministic; change them here, not by hand in the YAML):
- Candidates: every estimation method converged, n_stores >= MIN_STORES, stockout_day_rate in RATE_RANGE
  (censoring matters, yet method 1 still has data). No selection on which method fits better: that would
  bias the method comparison.
- SKUs: rank candidates by n_stores (desc), then sku_id; N = 1, 2, 5 take the top 1, 2, 5 (nested).
- Scale: lambda_env = s * lambda_base, one global s = TARGET_LAMBDA / lambda_base(top SKU, censored_mle).
  The source units are a normalized scale (coefficient unknown), so only ratios carry information. The N=1
  cell then matches the team default lambda = 4 (DP size unchanged), and the gap between methods is preserved.
- Season: the method's day-of-week factors, season[t mod 7] with t = 0 taken as Monday.
- Capacity: C = round(kappa * O_ref) (SPEC section 1.1), with O_ref measured in the censored_mle world (the
  true demand) and passed in as `o_ref` {N: value}; scripts/measure_oref.py measures it. Without `o_ref`, the old
  provisional rule round(kappa * 5 * sum(lambda_env of censored_mle)). Either way it is the same for every method:
  capacity is a property of the warehouse, not an estimate, and cross-evaluation (D13) requires equal capacity.
- Costs: the SPEC section 1 defaults (the dataset has no prices).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

C1_SCHEMA_VERSION = 1
METHODS = {"m1": "drop_stockout_days", "m2": "hourly_extrapolation", "m3": "censored_mle"}
SCALE_METHOD = "censored_mle"
MIN_STORES = 100
RATE_RANGE = (0.3, 0.6)
TARGET_LAMBDA = 4.0
CAPACITY_PER_UNIT_DEMAND = 5.0   # team provisional rule: 20 per SKU at lambda = 4
SPEC_COSTS = {"price": 10.0, "order_cost": 4.0, "holding_cost": 0.1, "expiry_cost": 2.0, "lost_sales_cost": 2.0}
SPEC_DYNAMICS = {"shelf_life": 3, "lead_time": 2, "horizon": 30}
REQUIRED = {"sku_id", "estimation_method", "lambda_base", "converged", "n_stores", "stockout_day_rate",
            "schema_version", "source", *(f"dow_factor_{d}" for d in range(7))}


def load_c1(path: str | Path) -> dict[int, dict[str, dict]]:
    """{sku_id: {estimation_method: row}}. Refuses files that are not contract C1 v1."""
    table = pq.read_table(path)
    missing = REQUIRED - set(table.column_names)
    if missing:
        raise ValueError(f"{path} is not a C1 demand_params file; missing columns {sorted(missing)}")
    out: dict[int, dict[str, dict]] = {}
    for row in table.to_pylist():
        if row["schema_version"] != C1_SCHEMA_VERSION:
            raise ValueError(f"C1 schema v{row['schema_version']}, this code reads v{C1_SCHEMA_VERSION}")
        out.setdefault(row["sku_id"], {})[row["estimation_method"]] = row
    return out


def select_skus(params: dict[int, dict[str, dict]], n: int) -> list[int]:
    """Top-n candidates by store count (ties: smaller sku_id). Raises if fewer than n qualify."""
    lo, hi = RATE_RANGE
    candidates = [
        (-next(iter(m.values()))["n_stores"], sku) for sku, m in params.items()
        if set(METHODS.values()) <= set(m) and all(r["converged"] for r in m.values())
        and next(iter(m.values()))["n_stores"] >= MIN_STORES
        and lo <= next(iter(m.values()))["stockout_day_rate"] <= hi
    ]
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} SKUs meet the selection rules, need {n}")
    return [sku for _, sku in sorted(candidates)[:n]]


def build_configs(params: dict[int, dict[str, dict]], ns=(1, 2, 5), kappas=(0.6, 1.0),
                  methods=("m1", "m3"), o_ref: dict[int, float] | None = None) -> dict[str, dict]:
    """{file stem: YAML dict}. Keys outside EnvConfig (calibration) are provenance only; the loader ignores them."""
    skus_all = select_skus(params, max(ns))
    scale = TARGET_LAMBDA / params[skus_all[0]][SCALE_METHOD]["lambda_base"]
    out = {}
    for n in ns:
        skus = skus_all[:n]
        true_total = sum(scale * params[s][SCALE_METHOD]["lambda_base"] for s in skus)
        for kappa in kappas:
            if o_ref is not None:
                capacity = max(1, round(kappa * o_ref[n]))
                capacity_rule = f"round(kappa * O_ref), O_ref = {o_ref[n]} ({SCALE_METHOD} world)"
            else:
                capacity = max(1, round(kappa * CAPACITY_PER_UNIT_DEMAND * true_total))
                capacity_rule = f"round(kappa * {CAPACITY_PER_UNIT_DEMAND} * sum(lambda_env, {SCALE_METHOD}))"
            for short in methods:
                rows = [params[s][METHODS[short]] for s in skus]
                lam = [round(scale * r["lambda_base"], 6) for r in rows]
                # one weekly pattern for the whole env (SPEC: season is shared): demand-weighted mean of factors
                season = [sum(r[f"dow_factor_{d}"] * x for r, x in zip(rows, lam, strict=True)) / sum(lam)
                          for d in range(7)]
                season = [round(v / (sum(season) / 7), 6) for v in season]
                out[f"frn_{short}_n{n}_k{str(kappa).replace('.', '')}"] = {
                    "n_skus": n, "kappa": kappa, "capacity": capacity, **SPEC_DYNAMICS,
                    "demand_mean": lam, "season": season, **SPEC_COSTS,
                    "calibration": {
                        "source": rows[0]["source"], "estimation_method": METHODS[short], "sku_ids": skus,
                        "lambda_base": [r["lambda_base"] for r in rows], "scale": round(scale, 9),
                        "scale_rule": f"{TARGET_LAMBDA} / lambda_base(sku {skus_all[0]}, {SCALE_METHOD})",
                        "capacity_rule": capacity_rule,
                        "season_t0": "Monday",
                    },
                }
    return out
