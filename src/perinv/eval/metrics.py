"""Report tables from the warehouse. Metric definitions live once, in eval/sql/views.sql (WAREHOUSE.md §5.1);
this module only reads the views and adds IQM + CIs over seeds.

    con = warehouse.connect("runs")
    run_table(con)                      # one row per run: profit, waste, fill, utilisation, % of DP, CPU hours
    summary(con)                        # one row per (group, N, kappa, policy): IQM over seeds + 95% CI
    paired_vs(con, "ppo", "ss")         # per-cell mean per-path profit difference, seed by seed
    paired_groups(con, "D13", "D3")     # cross-evaluation: same policy + seed, same eval world, group A - group B
    paired_worlds(con, "A_spike", "A")  # shock cost: same trained model, same path index, shocked world - base world

Owner: Member 3.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from perinv.eval.stats import bootstrap_ci, paired_difference

CELL = ["group", "n_skus", "kappa", "policy"]
METRICS = ["mean_profit", "pct_of_dp", "waste_rate", "fill_rate", "capacity_utilisation", "cpu_hours"]


def run_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.sql("""
        SELECT r.*, d.pct_of_dp, d.mean_gap_to_dp
        FROM v_run r LEFT JOIN v_vs_dp d USING (run_key)
        ORDER BY "group", n_skus, kappa, policy, seed
    """).df()


def summary(con: duckdb.DuckDBPyConnection, metrics: list[str] = METRICS, seed: int = 0) -> pd.DataFrame:
    """IQM over seeds with a percentile bootstrap CI, per cell and metric."""
    runs = run_table(con)
    rows = []
    for cell, g in runs.groupby(CELL, sort=True):
        row = dict(zip(CELL, cell, strict=True)) | {"n_seeds": len(g)}
        for m in metrics:
            est = bootstrap_ci(g[m].to_numpy(dtype=float), seed=seed)
            row |= {m: est.value, f"{m}_low": est.low, f"{m}_high": est.high}
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs(con: duckdb.DuckDBPyConnection, policy: str, baseline: str, seed: int = 0) -> pd.DataFrame:
    """For every run of `policy`: mean per-path profit difference to `baseline` on the same config and paths.

    Baselines run with one seed; if several exist, their per-path profits are averaged first.
    """
    pairs = con.execute("""
        WITH p AS (SELECT v.*, d.eval_config_hash, d.policy, d.seed, d."group", d.n_skus, d.kappa
                   FROM v_path v JOIN dim_run d USING (run_key)),
        b AS (SELECT eval_config_hash, path_id, AVG(profit) AS base FROM p WHERE policy = ? GROUP BY ALL)
        SELECT p."group", p.n_skus, p.kappa, p.seed, p.path_id, p.profit, b.base
        FROM p JOIN b USING (eval_config_hash, path_id)
        WHERE p.policy = ?
        ORDER BY ALL
    """, [baseline, policy]).df()
    rows = []
    for key, g in pairs.groupby(["group", "n_skus", "kappa", "seed"], sort=True):
        est = paired_difference(g["profit"], g["base"], seed=seed)
        rows.append(dict(zip(["group", "n_skus", "kappa", "seed"], key, strict=True))
                    | {"policy": policy, "baseline": baseline, "mean_diff": est.value,
                       "low": est.low, "high": est.high, "n_paths": est.n})
    return pd.DataFrame(rows)


def paired_groups(con: duckdb.DuckDBPyConnection, group: str, reference: str, seed: int = 0) -> pd.DataFrame:
    """Profit of `group` minus `reference`, per (policy, seed), paired on the same eval world and test path.

    Question 4: paired_groups(con, "D13", "D3") = what a policy built on method-1 demand loses, in the
    method-3 world, against the same policy built on method-3 demand. Negative = calibration cost.
    """
    pairs = con.execute("""
        WITH p AS (SELECT v.path_id, v.profit, d."group", d.policy, d.seed, d.n_skus, d.kappa, d.eval_config_hash
                   FROM v_path v JOIN dim_run d USING (run_key))
        SELECT a.policy, a.seed, a.n_skus, a.kappa, a.path_id, a.profit AS profit, b.profit AS ref_profit
        FROM p a JOIN p b USING (policy, seed, eval_config_hash, path_id)
        WHERE a."group" = ? AND b."group" = ?
        ORDER BY ALL
    """, [group, reference]).df()
    rows = []
    for key, g in pairs.groupby(["policy", "seed", "n_skus", "kappa"], sort=True):
        est = paired_difference(g["profit"], g["ref_profit"], seed=seed)
        rows.append(dict(zip(["policy", "seed", "n_skus", "kappa"], key, strict=True))
                    | {"group": group, "reference": reference, "mean_diff": est.value, "low": est.low,
                       "high": est.high, "mean_ref_profit": float(g["ref_profit"].mean()), "n_paths": est.n})
    return pd.DataFrame(rows)


def paired_worlds(con: duckdb.DuckDBPyConnection, group: str, reference: str, seed: int = 0) -> pd.DataFrame:
    """Profit of the same model (train_key) on path k of `group`'s eval world minus path k of `reference`'s.

    For demand shocks: paired_worlds(con, "A_spike", "A") = what the spike costs each policy. Shocked path k is
    base path k plus the shock (perinv.data.scenarios), so the per-path difference isolates the shock. The
    runner guarantees that pairing by refusing shock configs whose base demand differs from the build config.
    """
    pairs = con.execute("""
        WITH p AS (SELECT v.path_id, v.profit, d."group", d.policy, d.seed, d.n_skus, d.kappa, d.train_key
                   FROM v_path v JOIN dim_run d USING (run_key))
        SELECT a.policy, a.seed, a.n_skus, a.kappa, a.path_id, a.profit AS profit, b.profit AS ref_profit
        FROM p a JOIN p b USING (train_key, path_id)
        WHERE a."group" = ? AND b."group" = ?
        ORDER BY ALL
    """, [group, reference]).df()
    rows = []
    for key, g in pairs.groupby(["policy", "seed", "n_skus", "kappa"], sort=True):
        est = paired_difference(g["profit"], g["ref_profit"], seed=seed)
        rows.append(dict(zip(["policy", "seed", "n_skus", "kappa"], key, strict=True))
                    | {"group": group, "reference": reference, "mean_diff": est.value, "low": est.low,
                       "high": est.high, "mean_ref_profit": float(g["ref_profit"].mean()), "n_paths": est.n})
    return pd.DataFrame(rows)
