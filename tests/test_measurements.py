"""U4 measurement code: the provisional O_ref stand-in, the DP daily profile (cold start), the benchmark script."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fakes import tiny_config

from perinv.data.crn import generate_paths
from perinv.dp.backward_induction import build_table, daily_profile, evaluate_policy, solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_vecenv  # noqa: E402
import measure_oref  # noqa: E402

sys.path.pop(0)


def test_order_up_to_policy():
    p = measure_oref.OrderUpTo([10, 4])
    info = {"state": {"inv": np.array([[2, 1, 0], [5, 0, 0]]), "pipe": np.array([[3, 0], [0, 1]]), "t": 0}}
    assert p.act(None, info).tolist() == [4, 0]


def test_oref_is_the_sum_of_levels_and_skus_are_independent():
    """Order-up-to keeps the post-order position at exactly S, so O_ref = sum(S_i); identical SKUs get equal S."""
    cfg = replace(tiny_config(n_skus=2, capacity=40, horizon=20))
    demand = generate_paths(cfg, "tune", 20, 0)
    r = measure_oref.measure(cfg, demand)
    assert r.levels[0] == r.levels[1] and r.o_ref == pytest.approx(sum(r.levels))
    one = replace(cfg, n_skus=1, demand_mean=(4.0,), price=(10.0,), order_cost=(4.0,), holding_cost=(0.1,),
                  expiry_cost=(2.0,), lost_sales_cost=(2.0,))
    assert measure_oref.best_level(one, 0, demand[:, :, :1])[0] == r.levels[0]


def test_oref_grid_edge_fails_loud():
    cfg = tiny_config(n_skus=1, capacity=40, horizon=20)
    with pytest.raises(RuntimeError, match="grid edge"):
        measure_oref.best_level(cfg, 0, generate_paths(cfg, "tune", 5, 0), max_level=3)


def test_daily_profile_adds_up_to_the_value():
    cfg = replace(tiny_config(n_skus=1, capacity=10, horizon=12), season=(0.5, 1.5, 1, 1, 1, 1, 1))
    table = build_table(cfg)
    sol = solve(cfg, table)
    empty = np.zeros(len(sol.space))
    empty[sol.space.index(np.zeros(sol.space.dim, np.int64))] = 1
    prof = daily_profile(cfg, sol.pi, empty, table)
    assert prof.reward.sum() == pytest.approx(sol.V[0, empty.argmax()]) and prof.occupancy[0] == 0
    mix = np.full(len(sol.space), 1 / len(sol.space))          # any start distribution
    const = np.full_like(sol.pi, 3)
    assert daily_profile(cfg, const, mix, table).reward.sum() == pytest.approx(
        mix @ evaluate_policy(cfg, const, table)[0])


def test_cold_start_days_lose_money():
    """With an empty pipeline nothing arrives for tau days: those days are forced stockouts (D4 evidence)."""
    cfg = tiny_config(n_skus=1, capacity=12, horizon=30)
    sol = solve(cfg)
    empty = np.zeros(len(sol.space))
    empty[sol.space.index(np.zeros(sol.space.dim, np.int64))] = 1
    reward = daily_profile(cfg, sol.pi, empty).reward
    assert (reward[:cfg.lead_time] < 0).all() and reward[10:20].mean() > 0


def test_bench_raw_env_runs():
    assert bench_vecenv.raw_steps_per_second("configs/env/n1_k06.yaml", 200) > 0
