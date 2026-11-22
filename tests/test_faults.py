"""Fault injection: the pipeline either completes correctly or fails loud and contained, never silently wrong.

Each test names one failure mode. Builders and envs that misbehave live in tests/fakes.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fakes import BAD_ACTIONS, ConstantPolicy, FakeEnv, FixedActionPolicy, NanRewardEnv, tiny_config
from test_runner import ENV_BASE, n_done

from perinv.data.crn import generate_paths, write_paths
from perinv.env.config import load_env_config
from perinv.eval import metrics, warehouse
from perinv.eval.evaluate import InvalidActionError, as_order, evaluate
from perinv.eval.writer import EvalValidationError, validate
from perinv.runner import expand, plan, run_matrix
from perinv.utils.generated import publish
from perinv.utils.run_io import mark_crashed, read_manifest, run_status

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def ws(tmp_path):
    """N=1 base world, a full-horizon outage (demand 0 everywhere), and a 50x spike (demand far above C)."""
    cfgs = tmp_path / "configs"
    (cfgs / "env").mkdir(parents=True)
    base = ENV_BASE | {"n_skus": 1, "kappa": 1.0, "capacity": 12}
    variants = {
        "base": {},
        "dead": {"scenario": [{"kind": "outage", "start": 0, "duration": 7}]},
        "flood": {"scenario": [{"kind": "spike", "start": [0, 2], "duration": 5, "magnitude": 50.0}]},
    }
    for name, extra in variants.items():
        p = cfgs / "env" / f"{name}.yaml"
        p.write_text(yaml.safe_dump(base | extra))
        write_paths(load_env_config(p), tmp_path / "crn", root_seed=1, sizes={"tune": 2, "test": 5})
    (cfgs / "budget.yaml").write_text(yaml.safe_dump({"train_env_steps": 100, "seeds": [0], "profiles": {}}))
    return tmp_path


def env(ws, name):
    return (ws / "configs" / "env" / f"{name}.yaml").as_posix()


def jobs(ws, groups, builders, env_factory="fakes:FakeEnv", crn="crn"):
    m = ws / "matrix.yaml"
    m.write_text(yaml.safe_dump({"budget": (ws / "configs" / "budget.yaml").as_posix(), "env_factory": env_factory,
                                 "builders": builders, "groups": groups}))
    return expand(m, runs_root=(ws / "runs").as_posix(), crn_root=(ws / crn).as_posix(), configs_root=ws / "configs")


def failure(job) -> dict:
    return json.loads((job.run_dir / "failure.json").read_text())


# ---------------------------------------------------------------- malformed policy output

@pytest.mark.parametrize("kind", sorted(BAD_ACTIONS))
def test_malformed_action_is_rejected_with_path_and_day(kind):
    with pytest.raises(InvalidActionError, match="path 3, day 7"):
        as_order(BAD_ACTIONS[kind](2), 2, 3, 7)


@pytest.mark.parametrize("action", [np.int64(2), [2], np.array([2.0]), np.array([[2]])])
def test_valid_action_shapes_for_n1_are_accepted(action):
    assert as_order(action, 1, 0, 0).tolist() == [2]


def test_over_capacity_orders_are_left_to_the_env():
    """DQN relies on env clipping (SPEC §4.2), so evaluate must not reject large orders."""
    cfg = tiny_config(n_skus=1, capacity=5)
    res = evaluate(FixedActionPolicy(np.array([99])), FakeEnv(cfg), generate_paths(cfg, "test", 2, 0))
    assert res.clipped.any() and (res.qty["order"] <= 5).all()


def test_bad_actions_fail_their_run_only(ws):
    pols = [f"bad_{k}" for k in sorted(BAD_ACTIONS)] + ["ok"]
    js = jobs(ws, {"G": {"envs": [env(ws, "base")], "policies": pols}},
              {**{p: "fakes:build_bad_action" for p in pols[:-1]}, "ok": "fakes:build_constant"})
    assert run_matrix(js) == {"failed": len(BAD_ACTIONS), "done": 1, "skipped": 0}
    for j in js:
        if j.policy == "ok":
            assert run_status(j.run_dir, j.keys()[1]) == "done"
        else:
            f = failure(j)
            assert f["type"] == "InvalidActionError" and f["stage"] == "eval" and "path 0, day 0" in f["message"]


# ---------------------------------------------------------------- buggy env

def test_nan_reward_is_rejected_not_written(ws):
    js = jobs(ws, {"G": {"envs": [env(ws, "base")], "policies": ["ppo"]}}, {"ppo": "fakes:build_constant"},
              env_factory="fakes:NanRewardEnv")
    assert run_matrix(js)["failed"] == 1
    f = failure(js[0])
    assert f["type"] == "EvalValidationError" and "non-finite reward" in f["message"] and "day 1" in f["message"]
    assert not (js[0].run_dir / "eval_test.parquet").exists()


def test_validate_rejects_nan_reward_directly():
    cfg = tiny_config(n_skus=1)
    demand = generate_paths(cfg, "test", 2, 0)
    res = evaluate(ConstantPolicy([2]), NanRewardEnv(cfg), demand)
    with pytest.raises(EvalValidationError, match="non-finite"):
        validate(res, cfg, demand)


def test_fractional_info_quantity_is_rejected():
    class HalfSoldEnv(FakeEnv):
        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            return obs, r, term, trunc, info | {"sold": info["sold"] + 0.5}
    cfg = tiny_config(n_skus=1)
    with pytest.raises(ValueError, match="whole units"):
        evaluate(ConstantPolicy([2]), HalfSoldEnv(cfg), generate_paths(cfg, "test", 1, 0))


# ---------------------------------------------------------------- hard worker crash

def test_worker_crash_fails_only_that_job(ws):
    pols = ["ppo", "die", "a2c", "sac"]
    js = jobs(ws, {"G": {"envs": [env(ws, "base")], "policies": pols}},
              {"ppo": "fakes:build_constant", "a2c": "fakes:build_constant", "sac": "fakes:build_constant",
               "die": "fakes:build_die"})
    assert run_matrix(js, n_jobs=2) == {"done": 3, "failed": 1, "skipped": 0}
    dead = next(j for j in js if j.policy == "die")
    assert run_status(dead.run_dir, dead.keys()[1]) == "failed"
    f = failure(dead)
    assert f["type"] == "WorkerCrashed" and "exit code 3" in f["message"] and f["stage"] == "build"
    assert n_done([j for j in js if j.policy != "die"]) == 3
    assert run_matrix(js, n_jobs=2) == {"failed": 1, "skipped": 3}   # rerun: only the culprit runs again


def test_mark_crashed_never_overwrites_a_finished_run(ws):
    js = jobs(ws, {"G": {"envs": [env(ws, "base")], "policies": ["ppo"]}}, {"ppo": "fakes:build_constant"})
    run_matrix(js)
    j = js[0]
    assert not mark_crashed(j.run_dir, j.keys()[1], "boom")          # status is done: untouched
    assert not mark_crashed(j.run_dir, "other-run-key", "boom")      # another run's manifest: untouched
    assert read_manifest(j.run_dir)["status"] == "done" and not (j.run_dir / "failure.json").exists()


# ---------------------------------------------------------------- missing inputs

def test_missing_crn_lists_every_job_and_runs_nothing(ws):
    js = jobs(ws, {"G": {"envs": [env(ws, "base")], "policies": ["ppo", "a2c"]},
                   "H": {"envs": [env(ws, "base")], "eval_envs": [env(ws, "dead")], "policies": ["ppo"]}},
              {"ppo": "fakes:build_constant", "a2c": "fakes:build_constant"}, crn="empty")
    rows = plan(js)
    assert [a.split(":")[0] for _, _, a in rows] == ["no CRN"] * 3
    with pytest.raises(FileNotFoundError, match=r"3 job\(s\).*base\.yaml.*dead\.yaml"):
        run_matrix(js)
    assert not (ws / "runs").exists()


def test_dry_run_reports_missing_crn(ws):
    import subprocess
    js = jobs(ws, {"G": {"envs": [env(ws, "base")], "policies": ["ppo"]}}, {"ppo": "fakes:build_constant"})
    assert js
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "run_matrix.py"), str(ws / "matrix.yaml"),
                          "--dry-run", "--crn-root", str(ws / "empty"), "--configs-root", str(ws / "configs"),
                          "--runs-root", str(ws / "runs")],
                         capture_output=True, text=True, cwd=REPO, env={**__import__("os").environ,
                                                                       "PYTHONPATH": str(REPO / "tests")})
    assert res.returncode == 0, res.stderr
    assert "1 blocked (no CRN paths)" in res.stdout


# ---------------------------------------------------------------- extreme demand through the reporting layer

def test_extreme_demand_reaches_the_report(ws, tmp_path):
    """Demand 0 on every day (fill rate undefined everywhere) and demand far above capacity, end to end."""
    js = jobs(ws, {"B": {"envs": [env(ws, "base")], "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
                   "B_dead": {"envs": [env(ws, "base")], "eval_envs": [env(ws, "dead")],
                              "policies": ["ppo", {"name": "dp", "seeds": [0]}]},
                   "B_flood": {"envs": [env(ws, "base")], "eval_envs": [env(ws, "flood")],
                               "policies": ["ppo", {"name": "dp", "seeds": [0]}]}},
              {"ppo": "fakes:build_belief", "dp": "fakes:build_constant"})
    assert run_matrix(js)["done"] == 6
    con = warehouse.connect(ws / "runs")
    runs = metrics.run_table(con).set_index("run_name")
    assert runs.loc[["B_dead-N1-k1-ppo-s0", "B_dead-N1-k1-dp-s0"], "fill_rate"].isna().all()
    assert (runs.loc[["B_flood-N1-k1-ppo-s0"], "fill_rate"] < 0.5).all()
    summary = metrics.summary(con)
    dead = summary[summary.group == "B_dead"]
    assert dead.fill_rate.isna().all() and dead.mean_profit.notna().all()
    assert (metrics.paired_worlds(con, "B_dead", "B").mean_diff < 0).all()

    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import analyze
    finally:
        sys.path.remove(str(REPO / "scripts"))
    (tmp_path / "results").mkdir()
    figs = analyze.plot_profit(summary, tmp_path / "results")
    assert {p.name for p in figs} == {"profit_B.png", "profit_B_dead.png", "profit_B_flood.png"}


# ---------------------------------------------------------------- generated configs

HEADER = "# GENERATED by test\n"


def _ok(p):
    if "bad" in p.read_text():
        raise ValueError("bad file")


def test_publish_is_all_or_nothing(tmp_path):
    out = tmp_path / "out"
    publish({"a.yaml": "x: 1\n"}, out, header=HEADER, validate=_ok, owns=lambda p: True)
    with pytest.raises(ValueError, match="nothing was written"):
        publish({"a.yaml": "x: 2\n", "b.yaml": "bad\n"}, out, header=HEADER, validate=_ok, owns=lambda p: True)
    assert sorted(p.name for p in out.iterdir()) == ["a.yaml"]
    assert (out / "a.yaml").read_text() == HEADER + "x: 1\n"


def test_publish_removes_only_its_own_stale_files(tmp_path):
    out = tmp_path / "out"
    publish({"s__a.yaml": "1\n", "s__b.yaml": "1\n", "t__a.yaml": "1\n"}, out, header=HEADER, validate=_ok,
            owns=lambda p: True)
    (out / "s__hand.yaml").write_text("# written by a person\n")
    _, removed = publish({"s__a.yaml": "2\n"}, out, header=HEADER, validate=_ok,
                         owns=lambda p: p.name.startswith("s__"))
    assert [p.name for p in removed] == ["s__b.yaml"]
    assert sorted(p.name for p in out.iterdir()) == ["s__a.yaml", "s__hand.yaml", "t__a.yaml"]
    assert b"\r" not in (out / "s__a.yaml").read_bytes()


def test_shock_generator_drops_a_removed_preset(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import make_shock_configs as msc
    finally:
        sys.path.remove(str(REPO / "scripts"))
    base = REPO / "configs" / "env" / "n1_k06.yaml"
    msc.write_variants(base, tmp_path, msc.PRESETS)
    _, removed = msc.write_variants(base, tmp_path, {"spike": msc.PRESETS["spike"]})
    assert sorted(p.name for p in removed) == sorted(f"n1_k06__{k}.yaml" for k in msc.PRESETS if k != "spike")
