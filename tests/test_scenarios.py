"""Demand-shock scenarios (perinv.data.scenarios): eval-only, coupled to base paths, never silently ignored."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import yaml
from test_cross_eval import groups, matrix
from test_runner import ENV_BASE, jobs_for

from perinv.data.crn import generate_paths, write_paths
from perinv.data.scenarios import Shock, parse_scenario
from perinv.env.config import EnvConfig, load_env_config
from perinv.eval import metrics, warehouse
from perinv.runner import expand, run_matrix


def inline_config(n: int, capacity: int) -> EnvConfig:
    """The values of configs/env/n{1,5}_k06.yaml on 2026-11-22, inline so a teammate editing the provisional
    yaml does not break these tests. They guard the code (hashing, path generation), not the yaml."""
    return EnvConfig(n_skus=n, capacity=capacity, shelf_life=3, lead_time=2, horizon=30, demand_mean=(4.0,) * n,
                     season=(1.0,) * 7, price=(10.0,) * n, order_cost=(4.0,) * n, holding_cost=(0.1,) * n,
                     expiry_cost=(2.0,) * n, lost_sales_cost=(2.0,) * n, kappa=0.6, name=f"n{n}_k06")


BASE = inline_config(1, 12)
BASE5 = inline_config(5, 60)


def shocked(cfg, *shocks):
    return replace(cfg, scenario=tuple(shocks))


# ---------------------------------------------------------------- hashes and paths of existing configs

@pytest.mark.parametrize("cfg, demand_hash, config_hash, paths_sha", [
    (BASE, "1d13ace80597", "d2056241e64a", "ccbc63426c443d00"),
    (BASE5, "8a176507c0ba", "0bc3e42b407d", "0e4ae6f044543763"),
])
def test_configs_without_scenario_keep_hashes_and_paths(cfg, demand_hash, config_hash, paths_sha):
    """Values pinned before the scenario field existed: train/run keys and CRN files must not move."""
    assert (cfg.demand_hash(), cfg.config_hash()) == (demand_hash, config_hash)
    assert hashlib.sha256(generate_paths(cfg, "test", 5, 2026).tobytes()).hexdigest()[:16] == paths_sha


def test_scenario_changes_both_hashes():
    s = shocked(BASE, Shock("spike", (5, 10), 3, 3.0))
    assert s.demand_hash() != BASE.demand_hash() and s.config_hash() != BASE.config_hash()
    assert shocked(BASE, Shock("spike", (5, 10), 3, 2.0)).demand_hash() != s.demand_hash()


def test_every_generated_shock_config_differs_from_its_base():
    from pathlib import Path
    files = sorted(Path("configs/env/shocks").glob("*.yaml"))
    assert files
    for f in files:
        cfg = load_env_config(f)
        base = load_env_config(yaml.safe_load(f.read_text())["shock_of"])
        regen = "base config changed: rerun python scripts/make_shock_configs.py <base yaml>"
        assert cfg.scenario and cfg.demand_hash() != base.demand_hash(), f.name
        assert replace(cfg, scenario=(), name=base.name) == base, f"{f.name}: {regen}"


# ---------------------------------------------------------------- coupling and distributions

def test_shocked_path_equals_base_path_outside_the_shock():
    base = generate_paths(BASE, "test", 40, 7)
    for shock in [Shock("spike", (3, 20), 4, 3.0), Shock("drop", (3, 20), 4, 0.3), Shock("outage", (3, 20), 4)]:
        s = generate_paths(shocked(BASE, shock), "test", 40, 7)
        changed = (s != base).any(axis=2)                       # [path, day]
        for k in range(40):
            days = np.flatnonzero(changed[k])
            if days.size:
                assert days.max() - days.min() < 4               # only inside one 4-day window
        if shock.kind == "spike":
            assert (s >= base).all()
        else:
            assert (s <= base).all()
        if shock.kind == "outage":
            assert ((s == 0) | (s == base)).all()


def test_shock_paths_are_deterministic_and_independent_of_n_paths():
    cfg = shocked(BASE, Shock("overdispersed", (0, 0), 30, 0.5))
    a = generate_paths(cfg, "test", 10, 3)
    assert np.array_equal(a, generate_paths(cfg, "test", 10, 3))
    assert np.array_equal(a[:4], generate_paths(cfg, "test", 4, 3))
    assert not np.array_equal(generate_paths(cfg, "tune", 10, 3), a)


def test_shock_means_match_the_multiplier():
    lam = BASE.demand_mean[0]
    def window_mean(shock):
        return generate_paths(shocked(BASE, shock), "test", 2000, 11)[:, 10:15, 0].mean()
    assert window_mean(Shock("spike", (10, 10), 5, 3.0)) == pytest.approx(3 * lam, rel=0.05)
    assert window_mean(Shock("drop", (10, 10), 5, 0.4)) == pytest.approx(0.4 * lam, rel=0.05)
    assert window_mean(Shock("outage", (10, 10), 5)) == 0


def test_overdispersed_keeps_the_mean_and_adds_variance():
    lam, phi = BASE.demand_mean[0], 0.5
    d = generate_paths(shocked(BASE, Shock("overdispersed", (0, 0), 30, phi)), "test", 2000, 5)[:, :, 0]
    assert d.mean() == pytest.approx(lam, rel=0.05)
    assert d.var() == pytest.approx(lam + phi * lam**2, rel=0.1)    # negative binomial


def test_shock_on_selected_skus_only():
    cfg = BASE5
    base = generate_paths(cfg, "test", 20, 1)
    s = generate_paths(shocked(cfg, Shock("outage", (0, 5), 10, skus=(1, 3))), "test", 20, 1)
    assert np.array_equal(s[:, :, [0, 2, 4]], base[:, :, [0, 2, 4]])
    assert (s[:, :, [1, 3]] <= base[:, :, [1, 3]]).all()


# ---------------------------------------------------------------- invalid input fails loud

@pytest.mark.parametrize("raw, match", [
    ([{"kind": "spike", "start": 3, "duration": 2, "magnitude": 2, "magnitud": 3}], "unknown keys"),
    ([{"kind": "flood", "start": 3, "duration": 2}], "not one of"),
    ([{"kind": "spike", "start": 3, "duration": 2, "magnitude": 0.5}], "out of range"),
    ([{"kind": "drop", "start": 3, "duration": 2, "magnitude": 1.0}], "out of range"),
    ([{"kind": "outage", "start": 3, "duration": 2, "magnitude": 0.5}], "out of range"),
    ([{"kind": "spike", "start": [9, 3], "duration": 2, "magnitude": 2}], "lo <= hi"),
    ([{"kind": "spike", "start": 3, "duration": 0, "magnitude": 2}], "duration"),
    ([], "non-empty"),
])
def test_bad_shocks_are_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_scenario(raw)


def test_x_prefixed_keys_are_free_annotations(tmp_path):
    raw = yaml.safe_load(open("configs/env/n1_k06.yaml"))
    plain, annotated = tmp_path / "plain.yaml", tmp_path / "annotated.yaml"
    plain.write_text(yaml.safe_dump(raw))
    annotated.write_text(yaml.safe_dump(raw | {"x-note": "store 12, winter", "x-owner": {"member": "M3"}}))
    a, b = load_env_config(plain), load_env_config(annotated)
    assert (a.demand_hash(), a.config_hash()) == (b.demand_hash(), b.config_hash())
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw | {"note": "no prefix"}))
    with pytest.raises(ValueError, match="x-"):
        load_env_config(bad)


def test_shock_must_fit_the_env():
    with pytest.raises(ValueError, match="past the horizon"):
        shocked(BASE, Shock("spike", (5, 30), 2, 2.0))
    with pytest.raises(ValueError, match="targets SKU"):
        shocked(BASE, Shock("outage", (5, 6), 2, skus=(1,)))


def test_misspelt_scenario_key_is_not_silently_ignored(tmp_path):
    raw = yaml.safe_load(open("configs/env/n1_k06.yaml"))
    p = tmp_path / "typo.yaml"
    p.write_text(yaml.safe_dump(raw | {"scenarios": [{"kind": "outage", "start": 3, "duration": 2}]}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_env_config(p)


# ---------------------------------------------------------------- runner and warehouse

@pytest.fixture
def ws(tmp_path):
    """base (lambda 4) and its outage / spike variants, plus a 'drifted' variant whose base lambda changed."""
    cfgs = tmp_path / "configs"
    (cfgs / "env").mkdir(parents=True)
    base = ENV_BASE | {"n_skus": 1, "kappa": 1.0, "capacity": 40}
    variants = {"low": {}, "outage": {"scenario": [{"kind": "outage", "start": [0, 1], "duration": 5}]},
                "spike": {"scenario": [{"kind": "spike", "start": [0, 3], "duration": 3, "magnitude": 4.0}]},
                "drifted": {"demand_mean": 5.0, "scenario": [{"kind": "outage", "start": 1, "duration": 2}]}}
    for name, extra in variants.items():
        p = cfgs / "env" / f"{name}.yaml"
        p.write_text(yaml.safe_dump(base | extra))
        write_paths(load_env_config(p), tmp_path / "crn", root_seed=1, sizes={"tune": 2, "test": 6})
    (cfgs / "budget.yaml").write_text(yaml.safe_dump({"train_env_steps": 100, "seeds": [0, 1], "profiles": {}}))
    return tmp_path


SHOCK_GROUPS = {
    "L": {"envs": ["low"], "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
    "L_outage": {"envs": ["low"], "eval_envs": ["outage"], "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
    "L_spike": {"envs": ["low"], "eval_envs": ["spike"], "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
}


def test_scenario_config_cannot_be_trained_on(ws):
    bad = groups(ws, {"X": {"envs": ["outage"], "policies": ["ppo"]}})
    with pytest.raises(ValueError, match="eval-only"):
        jobs_for(ws, matrix(ws, bad))


def test_shock_eval_must_keep_the_base_demand(ws):
    bad = groups(ws, {"X": {"envs": ["low"], "eval_envs": ["drifted"], "policies": ["ppo"]}})
    with pytest.raises(ValueError, match="keep the base demand"):
        jobs_for(ws, matrix(ws, bad))


def test_paired_worlds_measures_the_shock_cost(ws):
    jobs = jobs_for(ws, matrix(ws, groups(ws, SHOCK_GROUPS)))
    assert run_matrix(jobs)["done"] == len(jobs) == 9
    for j in jobs:
        if j.group != "L":   # shock groups reuse the base model, they never train on the shocked world
            assert (j.run_dir / "builds.txt").read_text().split() == [f"reuse:{j.name.replace(j.group, 'L', 1)}"]
    con = warehouse.connect(ws / "runs")
    cost = metrics.paired_worlds(con, "L_outage", "L")
    assert set(cost.policy) == {"ppo", "dp"} and len(cost) == 3 and (cost.n_paths == 6).all()
    assert (cost.mean_diff < 0).all()              # closed store, same orders: less revenue, more waste
    assert len(metrics.paired_worlds(con, "L_spike", "L")) == 3
    assert metrics.paired_groups(con, "L_outage", "L").empty   # different eval worlds: that join must not pair
    # reporting path: shock groups appear in the tables, and % of DP is undefined there (DP is cross-evaluated)
    runs = metrics.run_table(con).set_index("run_name")
    assert runs.loc["L-N1-k1-ppo-s0", "pct_of_dp"] == pytest.approx(1.0)
    assert runs.loc[["L_outage-N1-k1-ppo-s0", "L_spike-N1-k1-dp-s0"], "pct_of_dp"].isna().all()
    assert set(metrics.summary(con).group) == {"L", "L_outage", "L_spike"}


def test_shock_matrix_expands():
    jobs = expand("configs/experiments/shocks.yaml", crn_root="data/crn")
    assert jobs and all(j.eval_env_path and j.eval_cfg.scenario and not j.cfg.scenario for j in jobs)
    assert {j.group for j in jobs} == {f"{a}_{s}" for a in "AB" for s in ("spike", "drop", "outage", "shift", "noisy")}
