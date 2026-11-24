"""run_io: keys, manifest lifecycle and status (docs/WAREHOUSE.md §4)."""

import json

import pytest
from fakes import ConstantPolicy, FakeEnv, tiny_config

from perinv.data.crn import generate_paths
from perinv.eval.evaluate import evaluate
from perinv.eval.writer import write_eval
from perinv.utils import run_io
from perinv.utils.run_io import (
    EVAL_FILE,
    DirtyTreeError,
    describe_env,
    installed_packages,
    read_manifest,
    run_context,
    run_key,
    run_name,
    run_status,
    train_key,
    train_reusable,
)

CFG = tiny_config()
HP = {"lr": 3e-4, "n_steps": 64}


def keys(seed=0, hp=HP, steps=1000, sha="aaa"):
    tk = train_key(CFG, policy="ppo", hparams=hp, seed=seed, train_env_steps=steps)
    return tk, run_key(tk, demand_hash=CFG.demand_hash(), crn_test_sha256=sha)


def meta():
    return describe_env(CFG) | {"group": "T", "policy": "ppo", "seed": 0, "crn_test_sha256": "aaa",
                                "train_env_steps": 1000}


def start(tmp_path, rk, tk, **kw):
    return run_context(tmp_path, "T-N2-k1-ppo-s0", run_key=rk, train_key=tk, config={"hp": HP}, meta=meta(), **kw)


def write_fact(run, rk):
    demand = generate_paths(CFG, "test", 3, root_seed=1)
    res = evaluate(ConstantPolicy([2, 2]), FakeEnv(CFG), demand)
    write_eval(res, CFG, demand, rk, run.dir / EVAL_FILE)


# ---------------------------------------------------------------- keys

def test_keys_change_only_with_inputs_that_change_results():
    tk, rk = keys()
    assert keys() == (tk, rk)                                    # deterministic
    assert keys(hp={"n_steps": 64, "lr": 3e-4}) == (tk, rk)      # dict order does not matter
    assert keys(seed=1)[0] != tk and keys(steps=2000)[0] != tk and keys(hp={"lr": 1e-3})[0] != tk
    tk2, rk2 = keys(sha="bbb")                                   # new test paths: re-evaluate, don't retrain
    assert tk2 == tk and rk2 != rk


def test_run_name_is_readable():
    assert run_name("A", 1, 0.6, "ppo", 0) == "A-N1-k0.6-ppo-s0"
    assert run_name("B", 5, 1.0, "sac", 4) == "B-N5-k1-sac-s4"


# ---------------------------------------------------------------- lifecycle

def test_success_path_writes_complete_manifest_and_logs(tmp_path):
    tk, rk = keys()
    with start(tmp_path, rk, tk) as run:
        assert read_manifest(run.dir)["status"] == "running"
        run.stage = "eval"
        write_fact(run, rk)
        run.update(actual_env_steps=1024)
        run.log("eval_done", rows=42)
    m = read_manifest(run.dir)
    assert m["status"] == "done" and m["run_key"] == rk and m["train_key"] == tk
    assert m["actual_env_steps"] == 1024 and m["cpu_seconds"] >= 0 and m["wall_seconds"] >= 0
    for k in run_io.REQUIRED_META + ("git_sha", "git_dirty", "pip_freeze", "start_time", "end_time"):
        assert k in m, k
    assert EVAL_FILE in m["artifacts"] and "config.yaml" in m["artifacts"]
    lines = [json.loads(x) for x in (run.dir / "log.jsonl").read_text().splitlines()]
    assert [x["event"] for x in lines] == ["start", "eval_done", "done"]
    assert all(x["run_key"] == rk for x in lines)
    assert not list(run.dir.glob("*.tmp")) and not (run.dir / "failure.json").exists()
    assert run_status(run.dir, rk) == "done"


@pytest.mark.parametrize("exc", [ValueError("NaN in policy loss"), KeyboardInterrupt()])
def test_failure_marks_failed_writes_failure_json_and_reraises(tmp_path, exc):
    tk, rk = keys()
    with pytest.raises(type(exc)), start(tmp_path, rk, tk) as run:
        run.stage = "train"
        raise exc
    m = read_manifest(run.dir)
    fail = json.loads((run.dir / "failure.json").read_text())
    assert m["status"] == "failed" and fail == m["error"]
    assert fail["stage"] == "train" and fail["type"] == type(exc).__name__ and fail["run_key"] == rk
    assert run_status(run.dir, rk) == "failed"


def test_missing_meta_is_rejected_before_anything_is_written(tmp_path):
    tk, rk = keys()
    with pytest.raises(ValueError, match="missing"):
        with run_context(tmp_path, "x", run_key=rk, train_key=tk, config={}, meta={"group": "T"}):
            pass
    assert list(tmp_path.iterdir()) == []


def test_official_run_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(run_io, "_git", lambda *a: " M src/x.py" if a[0] == "status" else "abc")
    tk, rk = keys()
    with pytest.raises(DirtyTreeError), start(tmp_path, rk, tk, official=True):
        pass
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- status

def test_status_states(tmp_path):
    tk, rk = keys()
    d = tmp_path / "T-N2-k1-ppo-s0"
    assert run_status(d, rk) == "missing"

    with start(tmp_path, rk, tk) as run:   # finished without an eval file: not done
        pass
    assert run_status(d, rk) == "stale"

    with start(tmp_path, rk, tk) as run:
        write_fact(run, rk)
    assert run_status(d, rk) == "done"
    assert run_status(d, keys(sha="bbb")[1]) == "stale"   # config changed -> key changed

    m = read_manifest(d)                                    # simulate a crash mid-run
    m["status"] = "running"
    (d / "manifest.json").write_text(json.dumps(m))
    assert run_status(d, rk) == "running"


def test_eval_file_from_another_run_is_not_done(tmp_path):
    tk, rk = keys()
    with start(tmp_path, rk, tk) as run:
        write_fact(run, "some-other-key")
    assert run_status(run.dir, rk) == "stale"


def test_finished_training_survives_a_stale_rerun_only_if_train_key_matches(tmp_path):
    tk, rk = keys()
    with start(tmp_path, rk, tk) as run:
        run.update(train={"train_key": tk, "status": "done", "actual_env_steps": 1024})
    assert train_reusable(run.dir, tk)

    _, rk_new = keys(sha="bbb")                            # only the test paths changed
    with start(tmp_path, rk_new, tk) as run:
        assert train_reusable(run.dir, tk)

    tk3, rk3 = keys(seed=3)                                # training inputs changed
    with start(tmp_path, rk3, tk3) as run:
        assert not train_reusable(run.dir, tk3) and read_manifest(run.dir)["train"] is None


def test_installed_packages_works_without_pip():
    pkgs = installed_packages()
    assert any(p.lower().startswith("numpy==") for p in pkgs)
    assert any(p.lower().startswith("duckdb==") for p in pkgs)


# ---------------------------------------------------------------- Windows sharing violations (regression)

def test_atomic_write_retries_when_a_reader_holds_the_file(tmp_path, monkeypatch):
    """Parallel workers read each other's manifests; on Windows os.replace then fails with PermissionError."""
    import os

    from perinv.utils import atomic

    real, calls = os.replace, {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky_replace)
    atomic.atomic_write_text(tmp_path / "m.json", "{}")
    assert (tmp_path / "m.json").read_text() == "{}" and calls["n"] == 3


def test_failure_while_finishing_is_recorded_not_left_running(tmp_path, monkeypatch):
    """Regression: the final 'done' write failed outside the error handler and left the run at 'running'."""
    tk, rk = keys()
    real_update = run_io.Run.update

    def update(self, **fields):
        if fields.get("status") == "done":
            raise PermissionError(32, "file in use by another process")
        return real_update(self, **fields)

    monkeypatch.setattr(run_io.Run, "update", update)
    with pytest.raises(PermissionError), start(tmp_path, rk, tk) as run:
        pass
    m = read_manifest(run.dir)
    assert m["status"] == "failed" and m["error"]["stage"] == "finish"
    assert (run.dir / "failure.json").exists()
