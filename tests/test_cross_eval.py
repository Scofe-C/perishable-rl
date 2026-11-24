"""Cross-evaluation: build a policy on config A, score it on config B's test paths (plan question 4)."""

import json

import pytest
import yaml
from test_runner import ENV_BASE, jobs_for

from perinv.data.crn import write_paths
from perinv.env.config import load_env_config
from perinv.eval import metrics, warehouse
from perinv.runner import expand, run_matrix
from perinv.utils.run_io import read_manifest, run_status


@pytest.fixture
def ws(tmp_path):
    """low (lambda 3) and high (lambda 6) worlds, identical otherwise; plus an incompatible one (other capacity)."""
    cfgs = tmp_path / "configs"
    (cfgs / "env").mkdir(parents=True)
    for name, lam, cap in [("low", 3.0, 40), ("high", 6.0, 40), ("high_bigC", 6.0, 60)]:
        p = cfgs / "env" / f"{name}.yaml"
        p.write_text(yaml.safe_dump(ENV_BASE | {"n_skus": 1, "kappa": 1.0, "capacity": cap, "demand_mean": lam}))
        write_paths(load_env_config(p), tmp_path / "crn", root_seed=1, sizes={"tune": 2, "test": 6})
    (cfgs / "budget.yaml").write_text(yaml.safe_dump({"train_env_steps": 100, "seeds": [0, 1], "profiles": {}}))
    return tmp_path


def env(ws, name):
    return (ws / "configs" / "env" / f"{name}.yaml").as_posix()


def matrix(ws, groups):
    m = {"budget": (ws / "configs" / "budget.yaml").as_posix(), "env_factory": "fakes:FakeEnv",
         "builders": {"ppo": "fakes:build_belief", "dp": "fakes:build_belief"}, "groups": groups}
    path = ws / "matrix.yaml"
    path.write_text(yaml.safe_dump(m))
    return path


GROUPS = {
    "L": {"envs": ["low"], "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
    "H": {"envs": ["high"], "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
    "LH": {"envs": ["low"], "eval_envs": ["high"], "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
}


def groups(ws, spec=GROUPS):
    return {g: {k: ([env(ws, e) for e in v] if k in ("envs", "eval_envs") else v) for k, v in d.items()}
            for g, d in spec.items()}


def test_cross_eval_jobs_share_train_key_but_not_run_key(ws):
    jobs = {j.name: j for j in jobs_for(ws, matrix(ws, groups(ws)))}
    native, cross = jobs["L-N1-k1-ppo-s0"], jobs["LH-N1-k1-ppo-s0"]
    assert cross.eval_env_path.endswith("high.yaml") and native.eval_env_path is None
    assert cross.keys()[0] == native.keys()[0]          # same model
    assert cross.keys()[1] != native.keys()[1]          # different evaluation
    assert cross.keys()[1] != jobs["H-N1-k1-ppo-s0"].keys()[1]


def test_incompatible_or_unpaired_eval_envs_are_rejected(ws):
    bad = groups(ws, {"X": {"envs": ["low"], "eval_envs": ["high_bigC"], "policies": ["ppo"]}})
    with pytest.raises(ValueError, match=r"differ in \['capacity'\]"):
        jobs_for(ws, matrix(ws, bad))
    unpaired = groups(ws, {"X": {"envs": ["low", "high"], "eval_envs": ["high"], "policies": ["ppo"]}})
    with pytest.raises(ValueError, match="pair 1:1"):
        jobs_for(ws, matrix(ws, unpaired))


def test_cross_eval_reuses_the_native_model_and_scores_in_the_eval_world(ws):
    m = matrix(ws, groups(ws))
    native = [j for j in jobs_for(ws, m) if j.group in ("L", "H")]
    assert run_matrix(native)["done"] == 6
    cross = [j for j in jobs_for(ws, m) if j.group == "LH"]
    assert run_matrix(cross)["done"] == 3
    for j in cross:
        assert (j.run_dir / "builds.txt").read_text().split() == [f"reuse:{j.name.replace('LH-', 'L-')}"]
        man = read_manifest(j.run_dir)
        assert man["cross_eval"] is True
        assert man["eval_config_hash"] == load_env_config(env(ws, "high")).config_hash()
        assert man["config_hash"] == load_env_config(env(ws, "low")).config_hash()
        assert run_status(j.run_dir, j.keys()[1]) == "done"
    log = [json.loads(x) for x in (cross[0].run_dir / "log.jsonl").read_text().splitlines()]
    assert any(x["event"] == "reuse_model" for x in log)


def test_paired_groups_measures_the_calibration_cost(ws):
    run_matrix(jobs_for(ws, matrix(ws, groups(ws))))
    con = warehouse.connect(ws / "runs")
    cost = metrics.paired_groups(con, "LH", "H")
    assert set(cost.policy) == {"ppo", "dp"} and (cost.n_paths == 6).all()
    assert (cost.mean_diff < 0).all()          # believing demand is 3 when it is 6 loses money
    # % of DP uses the native DP of the eval world as reference, never the cross-evaluated DP
    runs = metrics.run_table(con).set_index("run_name")
    assert runs.loc["H-N1-k1-ppo-s0", "pct_of_dp"] == pytest.approx(1.0)
    assert runs.loc["LH-N1-k1-ppo-s0", "pct_of_dp"] < 1.0
    assert bool(runs.loc["LH-N1-k1-ppo-s0", "cross_eval"]) and not bool(runs.loc["H-N1-k1-ppo-s0", "cross_eval"])


def test_calibrated_matrix_has_cross_eval_group():
    jobs = expand("configs/experiments/calibrated.yaml", crn_root="data/crn")
    cross = [j for j in jobs if j.eval_env_path]
    assert cross and all("frn_m1_" in j.env_path and "frn_m3_" in j.eval_env_path for j in cross)


def test_paired_vs_uses_only_the_native_baseline(ws, monkeypatch):
    """Review bug: a cross-evaluated baseline (LH dp, built for `low`) was averaged into the `high` reference."""
    monkeypatch.chdir(ws)
    run_matrix(jobs_for(ws, matrix(ws, groups(ws))))
    con = warehouse.connect(ws / "runs")
    diff = metrics.paired_vs(con, "ppo", "dp").set_index(["group", "seed"])["mean_diff"]
    # fakes:build_belief gives ppo and dp the same policy, so a native comparison is exactly 0
    assert diff.loc[("H", 0)] == pytest.approx(0.0) and diff.loc[("L", 0)] == pytest.approx(0.0)


def test_pairing_ignores_runs_scored_on_other_test_paths(ws, monkeypatch):
    """Review bug: after regenerating the test paths and rerunning only ppo, path k of ppo was paired with a
    different demand path k of the stale dp run. Now such runs do not pair at all."""
    monkeypatch.chdir(ws)
    m = matrix(ws, groups(ws, {"L": GROUPS["L"]}))
    run_matrix(jobs_for(ws, m))
    write_paths(load_env_config(env(ws, "low")), ws / "crn", root_seed=7, sizes={"tune": 2, "test": 6})
    run_matrix([j for j in jobs_for(ws, m) if j.policy == "ppo"])
    con = warehouse.connect(ws / "runs")
    assert metrics.run_table(con).query("policy == 'ppo'")["pct_of_dp"].isna().all()
    assert metrics.paired_vs(con, "ppo", "dp").empty
    run_matrix([j for j in jobs_for(ws, m) if j.policy == "dp"])   # rerun dp on the new paths: pairs again
    con = warehouse.connect(ws / "runs")
    assert metrics.run_table(con).query("policy == 'ppo'")["pct_of_dp"].tolist() == pytest.approx([1.0, 1.0])
