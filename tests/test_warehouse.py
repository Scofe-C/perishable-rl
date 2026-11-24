"""Warehouse reader, SQL views, metrics, stats and analyze.py (docs/WAREHOUSE.md §5 and §6)."""

import json
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_runner import REPO, TESTS, envs, jobs_for, write_matrix
from test_runner import ws as ws  # noqa: F401  (fixture)

from perinv.eval import metrics, warehouse
from perinv.eval.stats import bootstrap_ci, iqm, paired_difference
from perinv.runner import run_matrix
from perinv.utils.run_io import EVAL_FILE, read_manifest

PATHS, T = 4, 7  # test_runner workspace: 4 test paths, horizon 7


@pytest.fixture
def filled(ws):  # noqa: F811
    """N=1 (kappa 0.6) and N=2 (kappa 1.0); ppo has 2 seeds, dp and ss one."""
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06", "n2_k10"),
                                "policies": ["ppo", {"name": "dp", "seeds": [0]}, {"name": "ss", "seeds": [0]}],
                                "seeds": [0, 1]}},
                     builders={"ss": "fakes:build_random"})
    jobs = jobs_for(ws, m)
    assert run_matrix(jobs)["done"] == len(jobs) == 8
    return ws, jobs


def q(con, sql):
    return con.execute(sql).fetchall()


def test_types_are_fixed_and_filters_count_correctly(filled):
    ws, _ = filled
    con = warehouse.connect(ws / "runs")
    types = dict(q(con, "SELECT column_name, column_type FROM (DESCRIBE dim_run)"))
    assert types["kappa"] == "DOUBLE" and types["n_skus"] == "INTEGER" and types["price"] == "DOUBLE[]"
    # the spike bug: kappa = 1 must not match the 0.6 runs
    rows_k1 = q(con, "SELECT count(*) FROM fact_step JOIN dim_run USING (run_key) WHERE kappa = 1")[0][0]
    assert rows_k1 == 4 * PATHS * T * 2          # 4 runs at N=2
    assert q(con, "SELECT count(*) FROM dim_run WHERE n_skus = 1")[0][0] == 4


def test_money_reconciles_with_env_reward(filled):
    ws, _ = filled
    con = warehouse.connect(ws / "runs")
    gap = q(con, """SELECT max(abs(r - dr)) FROM (
                      SELECT run_key, path_id, t, sum(sku_reward) AS r, sum(day_reward) AS dr
                      FROM v_step GROUP BY ALL)""")[0][0]
    assert gap < 1e-9


def test_v_run_matches_what_the_runner_logged(filled):
    ws, jobs = filled
    con = warehouse.connect(ws / "runs")
    sql_profit = dict(q(con, "SELECT run_name, mean_profit FROM v_run"))
    for j in jobs:
        logged = [json.loads(x) for x in (j.run_dir / "log.jsonl").read_text().splitlines()]
        eval_line = next(x for x in logged if x["event"] == "eval_done")
        assert sql_profit[j.name] == pytest.approx(eval_line["mean_profit"])


def test_metric_ranges_and_pct_of_dp(filled):
    ws, _ = filled
    runs = metrics.run_table(warehouse.connect(ws / "runs"))
    assert len(runs) == 8 and (runs.n_paths == PATHS).all()
    for col in ("waste_rate", "fill_rate", "capacity_utilisation"):
        assert runs[col].between(0, 1).all(), col
    # fakes: ppo and dp use the same constant policy -> exactly 100% of DP; DP itself is 100% too
    assert np.allclose(runs.loc[runs.policy.isin(["ppo", "dp"]), "pct_of_dp"], 1.0)
    assert runs.loc[runs.policy == "ss", "pct_of_dp"].notna().all()


def test_only_done_runs_are_visible(filled):
    ws, jobs = filled
    crashed, running = jobs[0].run_dir, jobs[1].run_dir
    m = read_manifest(crashed) | {"status": "failed"}
    (crashed / "manifest.json").write_text(json.dumps(m))
    m = read_manifest(running) | {"status": "running"}
    (running / "manifest.json").write_text(json.dumps(m))
    con = warehouse.connect(ws / "runs")
    names = {r[0] for r in q(con, "SELECT run_name FROM dim_run")}
    assert len(names) == 6 and jobs[0].name not in names and jobs[1].name not in names
    assert q(con, "SELECT count(DISTINCT run_key) FROM fact_step")[0][0] == 6


def test_mixed_schema_versions_are_refused(filled):
    ws, jobs = filled
    f = jobs[0].run_dir / EVAL_FILE
    t = pq.read_table(f)
    meta = dict(t.schema.metadata) | {b"perinv.eval_schema_version": b"999"}
    pq.write_table(t.replace_schema_metadata(meta), f)
    with pytest.raises(warehouse.WarehouseError, match="schema v999"):
        warehouse.connect(ws / "runs")


def test_empty_warehouse_has_all_views(tmp_path):
    con = warehouse.connect(tmp_path)
    for view in ("dim_run", "fact_step", "v_step", "v_path", "v_run", "v_vs_dp"):
        assert q(con, f"SELECT count(*) FROM {view}") == [(0,)]


def test_summary_and_paired(filled):
    ws, _ = filled
    con = warehouse.connect(ws / "runs")
    s = metrics.summary(con)
    ppo = s[(s.policy == "ppo") & (s.n_skus == 1)].iloc[0]
    assert ppo.n_seeds == 2 and ppo.mean_profit_low <= ppo.mean_profit <= ppo.mean_profit_high
    p = metrics.paired_vs(con, "ppo", "dp")
    assert len(p) == 4 and np.allclose(p.mean_diff, 0) and (p.n_paths == PATHS).all()


# ---------------------------------------------------------------- stats

def test_iqm_drops_the_tails():
    assert iqm([1, 2, 3, 4, 100]) == pytest.approx(3.0)       # trims 1 value at each end
    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    x = [10.0, 12.0, 11.0, 30.0, 9.0]
    a, b = bootstrap_ci(x, seed=0), bootstrap_ci(x, seed=0)
    assert a == b and a.low <= a.value <= a.high and a.n == 5
    one = bootstrap_ci([7.0])
    assert one.value == one.low == one.high == 7.0


def test_paired_difference_uses_per_path_differences():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 30, size=100)                      # large path-to-path noise
    est = paired_difference(base + 1.0, base)                 # constant +1 improvement
    assert est.value == pytest.approx(1.0)
    assert est.low == pytest.approx(1.0) and est.high == pytest.approx(1.0)  # path noise cancels out
    with pytest.raises(ValueError, match="align"):
        paired_difference([1, 2], [1, 2, 3])


# ---------------------------------------------------------------- CLI

def test_analyze_cli_writes_tables_and_figure(filled, tmp_path):
    ws, _ = filled
    out = tmp_path / "results"
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "analyze.py"), "--runs-root", str(ws / "runs"),
                          "--out", str(out)], capture_output=True, text=True, timeout=120,
                         env={**__import__("os").environ, "PYTHONPATH": str(TESTS)})
    assert res.returncode == 0, res.stderr
    assert (out / "runs.csv").exists() and (out / "summary.csv").exists() and (out / "profit_A.png").exists()
