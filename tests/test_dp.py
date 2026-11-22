"""Exact N = 1 DP: state space, exactness against brute force, agreement with the env, optimality, and the
runner builder with its cache."""

from dataclasses import replace
from functools import cache
from math import comb

import numpy as np
import pytest
import yaml
from fakes import tiny_config
from scipy.stats import poisson
from test_runner import ENV_BASE

from perinv.baselines import dp_policy
from perinv.data.crn import generate_paths, write_paths
from perinv.dp import state_space
from perinv.dp.backward_induction import build_table, demand_terms, evaluate_policy, solve
from perinv.env.config import load_env_config
from perinv.env.dynamics import step
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.eval import metrics, warehouse
from perinv.eval.evaluate import evaluate
from perinv.runner import expand, run_matrix
from perinv.utils.run_io import read_manifest


def cfg1(capacity=12, L=3, tau=2, T=30, lam=4.0, season=(1.0,) * 7):
    return replace(tiny_config(n_skus=1, capacity=capacity, horizon=T), shelf_life=L, lead_time=tau,
                   demand_mean=(lam,), season=season)


# ---------------------------------------------------------------- state space

@pytest.mark.parametrize("C, L, P", [(12, 3, 2), (20, 3, 2), (5, 1, 1), (4, 4, 3), (0, 2, 2)])
def test_state_count_and_index(C, L, P):
    sp = state_space.build(C, L, P)
    assert len(sp) == comb(C + L - 1 + P, L - 1 + P)
    assert (np.diff(sp.keys) > 0).all() and (sp.states.sum(axis=1) <= C).all()
    assert np.array_equal(sp.index(sp.states), np.arange(len(sp)))
    inv, pipe = sp.to_env(sp.states)
    assert np.array_equal(sp.from_env(inv[:, 0], pipe[:, 0]), sp.states)


def test_index_rejects_non_states():
    sp = state_space.build(5, 3, 2)
    with pytest.raises(KeyError, match="not a DP state"):
        sp.index(np.array([5, 1, 0, 0]))
    with pytest.raises(ValueError, match="inv\\[L-1\\]"):
        sp.from_env(np.array([0, 0, 1]), np.array([0, 0]))


def test_demand_terms():
    pmf, tail, excess = demand_terms(3.7, 15)
    d = np.arange(200)
    for i in range(16):
        assert tail[i] == pytest.approx(poisson.pmf(d[d >= i], 3.7).sum())
        assert excess[i] == pytest.approx((np.maximum(d - i, 0) * poisson.pmf(d, 3.7)).sum())


# ---------------------------------------------------------------- exactness

def brute_force(cfg, d_max=40):
    """Independent expectimax over raw env states: no state index, no tail lumping (demand cut at d_max)."""
    lam_t = cfg.demand_mean[0] * np.asarray(cfg.season)[np.arange(cfg.horizon) % 7]
    C = cfg.capacity

    @cache
    def V(t, inv, pipe):
        if t == cfg.horizon:
            return 0.0
        occ = sum(inv) + sum(pipe)
        p = poisson.pmf(np.arange(d_max + 1), lam_t[t])
        best = -np.inf
        for q in range(C - occ + 1):
            tr = step(np.array([[inv]] * (d_max + 1)), np.array([[pipe]] * (d_max + 1)),
                      np.full((d_max + 1, 1), q), np.arange(d_max + 1)[:, None], cfg)
            val = sum(p[d] * (tr.reward[d] + V(t + 1, tuple(tr.inv[d, 0]), tuple(tr.pipe[d, 0])))
                      for d in range(d_max + 1))
            best = max(best, val)
        return best

    return V


@pytest.mark.parametrize("cfg", [
    cfg1(capacity=4, L=2, tau=1, T=4, lam=1.5),
    cfg1(capacity=3, L=3, tau=2, T=3, lam=1.2),
    cfg1(capacity=4, L=2, tau=1, T=5, lam=1.5, season=(0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
])
def test_dp_equals_brute_force(cfg):
    sol = solve(cfg)
    V = brute_force(cfg)
    inv, pipe = sol.space.to_env(sol.space.states)
    for s in range(len(sol.space)):
        assert sol.V[0, s] == pytest.approx(V(0, tuple(inv[s, 0]), tuple(pipe[s, 0])), abs=1e-9)


def test_table_rows_match_the_env():
    """Pitfall guard (conformance): a DP row's next state and reward are what the env does from that state."""
    cfg = cfg1(capacity=8)
    table = build_table(cfg)
    rng = np.random.default_rng(0)
    for r in rng.choice(len(table.row_pair), 300, replace=False):
        s, q, d = table.pair_state[table.row_pair[r]], table.pair_order[table.row_pair[r]], table.row_demand[r]
        env = PerishableInventoryEnv(cfg)
        env.reset(options={"demand_path": np.full((cfg.horizon, 1), d, np.int64)})
        inv, pipe = table.space.to_env(table.space.states[[s]])
        env.inv, env.pipe = inv[0], pipe[0]
        _, reward, _, _, info = env.step(np.array([q]))
        nxt = table.space.from_env(info["state"]["inv"][0], info["state"]["pipe"][0])
        assert table.space.index(nxt) == table.row_next[r] and reward == pytest.approx(table.row_reward[r])


def test_dp_beats_every_constant_order_policy():
    cfg = cfg1(capacity=12)
    table = build_table(cfg)
    sol = solve(cfg, table)
    assert np.array_equal(evaluate_policy(cfg, sol.pi, table), sol.V)
    for q in range(cfg.capacity + 1):
        const = evaluate_policy(cfg, np.full_like(sol.pi, q), table)
        assert (sol.V[0] >= const[0] - 1e-9).all()


@pytest.mark.parametrize("path", ["configs/env/n1_k06.yaml", "configs/env/calibrated/frn_m3_n1_k06.yaml"])
def test_value_matches_simulated_dp_policy(path):
    """Acceptance: V[0, empty] equals the DP policy's mean profit on simulated paths (seasonal, fractional λ too)."""
    cfg = load_env_config(path)
    sol = solve(cfg)
    demand = generate_paths(cfg, "tune", 800, 11)
    profit = evaluate(dp_policy.DPPolicy(sol), PerishableInventoryEnv(cfg), demand).per_path()["profit"]
    se = profit.std(ddof=1) / np.sqrt(len(profit))
    assert abs(profit.mean() - sol.initial_value()) < 4 * se


def test_n_greater_than_one_is_rejected():
    with pytest.raises(ValueError, match="N = 1"):
        build_table(tiny_config(n_skus=2))


# ---------------------------------------------------------------- builder and cache

def test_cache_roundtrip_and_invalidation(tmp_path):
    cfg = cfg1(capacity=6, T=10)
    sol, meta = dp_policy.load_or_solve(cfg, tmp_path)
    assert not meta["hit"]
    again, meta2 = dp_policy.load_or_solve(cfg, tmp_path)
    assert meta2["hit"] and np.array_equal(again.pi, sol.pi) and np.array_equal(again.V, sol.V)
    path = dp_policy.cache_path(cfg, tmp_path)
    with open(path, "wb") as fh:   # a file under the right name but for another config
        np.savez(fh, V=np.zeros((3, 3)), pi=np.zeros((2, 3), np.int16), config_hash="other", event_order_version="1")
    _, meta3 = dp_policy.load_or_solve(cfg, tmp_path)
    assert not meta3["hit"] and not list(tmp_path.glob("*.tmp"))


def test_unreadable_cache_is_a_miss(tmp_path):
    cfg = cfg1(capacity=6, T=10)
    path = dp_policy.cache_path(cfg, tmp_path)
    tmp_path.mkdir(exist_ok=True)
    path.write_bytes(b"not an npz, e.g. a file cut off by a crash")
    sol, meta = dp_policy.load_or_solve(cfg, tmp_path)
    assert not meta["hit"] and np.array_equal(sol.V, solve(cfg).V)
    assert dp_policy.load_or_solve(cfg, tmp_path)[1]["hit"]


def test_parallel_dp_jobs_share_one_cache_file(tmp_path, monkeypatch):
    """Several groups build the DP for the same config at once (shocks.yaml A_*): every job must finish."""
    monkeypatch.setenv("PERINV_DP_CACHE", str(tmp_path / "dpcache"))   # inherited by the spawned workers
    (tmp_path / "env").mkdir()
    p = tmp_path / "env" / "n1.yaml"
    p.write_text(yaml.safe_dump(ENV_BASE | {"n_skus": 1, "kappa": 1.0, "capacity": 16, "horizon": 14}))
    write_paths(load_env_config(p), tmp_path / "crn", root_seed=1, sizes={"tune": 2, "test": 3})
    (tmp_path / "budget.yaml").write_text(yaml.safe_dump({"train_env_steps": 10, "seeds": [0], "profiles": {}}))
    m = tmp_path / "m.yaml"
    groups = {g: {"envs": [p.as_posix()], "policies": [{"name": "dp", "seeds": [0]}]} for g in ("A", "B", "C", "D")}
    m.write_text(yaml.safe_dump({"budget": (tmp_path / "budget.yaml").as_posix(), "groups": groups}))
    jobs = expand(m, runs_root=(tmp_path / "runs").as_posix(), crn_root=(tmp_path / "crn").as_posix(),
                  configs_root=tmp_path)
    assert run_matrix(jobs, n_jobs=4)["done"] == 4
    assert [f.name for f in (tmp_path / "dpcache").iterdir()] == [f"{load_env_config(p).config_hash()}.npz"]


def test_dp_job_through_the_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("PERINV_DP_CACHE", str(tmp_path / "dpcache"))
    (tmp_path / "env").mkdir()
    p = tmp_path / "env" / "n1.yaml"
    p.write_text(yaml.safe_dump(ENV_BASE | {"n_skus": 1, "kappa": 1.0, "capacity": 10, "horizon": 14}))
    write_paths(load_env_config(p), tmp_path / "crn", root_seed=1, sizes={"tune": 2, "test": 20})
    (tmp_path / "budget.yaml").write_text(yaml.safe_dump({"train_env_steps": 10, "seeds": [0], "profiles": {}}))
    m = tmp_path / "m.yaml"
    m.write_text(yaml.safe_dump({"budget": (tmp_path / "budget.yaml").as_posix(),
                                 "builders": {"const": "fakes:build_constant"},
                                 "groups": {"A": {"envs": [p.as_posix()],
                                                  "policies": [{"name": "dp", "seeds": [0]},
                                                               {"name": "const", "seeds": [0]}]}}}))
    jobs = expand(m, runs_root=(tmp_path / "runs").as_posix(), crn_root=(tmp_path / "crn").as_posix(),
                  configs_root=tmp_path)
    assert run_matrix(jobs)["done"] == 2
    dp_job = next(j for j in jobs if j.policy == "dp")
    man = read_manifest(dp_job.run_dir)
    assert man["dp"]["hit"] is False and man["dp"]["value_empty_start"] > 0
    runs = metrics.run_table(warehouse.connect(tmp_path / "runs")).set_index("policy")
    assert runs.loc["dp", "pct_of_dp"] == pytest.approx(1.0) and runs.loc["const", "pct_of_dp"] < 1.0
