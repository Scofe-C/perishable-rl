"""Runner: matrix expansion, skip-if-done, failure isolation, model reuse, parallel and kill/resume."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from perinv.data.crn import write_paths
from perinv.env.config import load_env_config
from perinv.runner import expand, plan, run_matrix
from perinv.utils.run_io import run_status

TESTS = Path(__file__).parent
REPO = TESTS.parent
ENV_BASE = dict(shelf_life=3, lead_time=2, horizon=7, demand_mean=4.0, price=10.0, order_cost=4.0,
                holding_cost=0.1, expiry_cost=2.0, lost_sales_cost=2.0)


@pytest.fixture
def ws(tmp_path):
    """A throwaway workspace: 2 env configs, CRN paths, budget, tuned hparams and a matrix file."""
    cfgs = tmp_path / "configs"
    for name, n, kappa in [("n1_k06", 1, 0.6), ("n2_k10", 2, 1.0)]:
        p = cfgs / "env" / f"{name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(ENV_BASE | {"n_skus": n, "kappa": kappa, "capacity": 12 * n}))
        write_paths(load_env_config(p), tmp_path / "crn", root_seed=1, sizes={"tune": 2, "test": 4})
    (cfgs / "budget.yaml").write_text(yaml.safe_dump(
        {"train_env_steps": 500, "seeds": [0, 1, 2], "profiles": {"tiny": {"train_env_steps": 50}}}))
    (cfgs / "algo").mkdir()
    (cfgs / "algo" / "ppo.yaml").write_text(yaml.safe_dump({"hparams": {"lr": 0.001}}))
    (cfgs / "tuned").mkdir()
    (cfgs / "tuned" / "ppo-n2_k10.yaml").write_text(yaml.safe_dump({"hparams": {"lr": 0.0003}}))
    return tmp_path


def write_matrix(ws, groups, builders=None):
    m = {"budget": (ws / "configs" / "budget.yaml").as_posix(), "env_factory": "fakes:FakeEnv",
         "builders": {"ppo": "fakes:build_constant", "dp": "fakes:build_constant"} | (builders or {}),
         "groups": groups}
    path = ws / "matrix.yaml"
    path.write_text(yaml.safe_dump(m))
    return path


def envs(ws, *names):
    return [(ws / "configs" / "env" / f"{n}.yaml").as_posix() for n in names]


def jobs_for(ws, matrix, **kw):
    return expand(matrix, runs_root=(ws / "runs").as_posix(), crn_root=(ws / "crn").as_posix(),
                  configs_root=ws / "configs", **kw)


def n_done(jobs):
    return sum(run_status(j.run_dir, j.keys()[1]) == "done" for j in jobs)


# ---------------------------------------------------------------- expansion

def test_expand_counts_filters_seeds_and_hparams(ws):
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06", "n2_k10"),
                                "policies": ["ppo", {"name": "dp", "seeds": [0]}]}})
    jobs = jobs_for(ws, m)
    assert len(jobs) == 2 * 3 + 2 * 1                        # ppo: budget seeds, dp: pinned seed
    assert len(jobs_for(ws, m, seeds=[0, 1])) == 2 * 2 + 2   # --seeds does not touch pinned dp
    assert {j.policy for j in jobs_for(ws, m, policies={"dp"})} == {"dp"}
    assert jobs_for(ws, m, groups={"B"}) == []
    assert {j.train_env_steps for j in jobs_for(ws, m, profile="tiny")} == {50}
    hp = {j.cfg.n_skus: j.hparams for j in jobs if j.policy == "ppo"}
    assert hp == {1: {"lr": 0.001}, 2: {"lr": 0.0003}}       # tuned file beats the default
    assert jobs[0].name == "A-N1-k0.6-ppo-s0"


def test_expand_rejects_two_jobs_in_one_folder(ws):
    e = envs(ws, "n1_k06")
    m = write_matrix(ws, {"A": {"envs": e + e, "policies": ["ppo"]}})
    with pytest.raises(ValueError, match="same run directory"):
        jobs_for(ws, m)


def test_missing_crn_paths_fail_early(ws):
    """plan() lists every job without CRN paths; run_matrix refuses to start and names all of them."""
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06"), "policies": ["ppo"]}})
    jobs = expand(m, runs_root=(ws / "runs").as_posix(), crn_root=(ws / "nope").as_posix(),
                  configs_root=ws / "configs")
    assert all(action.startswith("no CRN") and "make_crn_paths" in action for _, _, action in plan(jobs))
    with pytest.raises(FileNotFoundError, match="make_crn_paths"):
        run_matrix(jobs)
    assert not (ws / "runs").exists()   # nothing ran


def test_unimplemented_builder_is_blocked_not_run(ws):
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06"), "policies": [{"name": "sac", "seeds": [0]}]}},
                     builders={"sac": "perinv.train.train:build"})
    (row,) = plan(jobs_for(ws, m))
    assert row[2].startswith("no builder") and "perinv.train.train:build" in row[2]
    assert run_matrix(jobs_for(ws, m)) == {"skipped": 1}


# ---------------------------------------------------------------- running

def test_run_then_rerun_skips_everything(ws):
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06", "n2_k10"), "policies": ["ppo"], "seeds": [0, 1]}})
    jobs = jobs_for(ws, m)
    assert run_matrix(jobs)["done"] == 4
    assert n_done(jobs) == 4
    assert run_matrix(jobs) == {"skipped": 4}
    assert run_matrix(jobs, force=True)["done"] == 4


def test_one_failure_does_not_stop_the_others(ws):
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06"), "policies": ["ppo", "dp"], "seeds": [0]}},
                     builders={"dp": "fakes:build_crash"})
    jobs = jobs_for(ws, m)
    counts = run_matrix(jobs)
    assert counts["done"] == 1 and counts["failed"] == 1
    bad = next(j for j in jobs if j.policy == "dp")
    fail = json.loads((bad.run_dir / "failure.json").read_text())
    assert fail["stage"] == "build" and fail["type"] == "FloatingPointError"
    assert run_matrix(jobs) == {"skipped": 1, "failed": 1}   # rerun retries only the failed one


def test_new_test_paths_reevaluate_without_retraining(ws):
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06"), "policies": ["ppo"], "seeds": [0]}})
    (job,) = jobs_for(ws, m)
    run_matrix([job])
    write_paths(job.cfg, ws / "crn", root_seed=99, sizes={"tune": 2, "test": 4})  # test.npz changes -> run_key
    (job,) = jobs_for(ws, m)
    assert run_status(job.run_dir, job.keys()[1]) == "stale"
    assert run_matrix([job])["done"] == 1
    assert (job.run_dir / "builds.txt").read_text().split() == ["train", f"reuse:{job.name}"]


def test_parallel_workers(ws):
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06", "n2_k10"), "policies": ["ppo"], "seeds": [0, 1]}})
    jobs = jobs_for(ws, m)
    assert run_matrix(jobs, n_jobs=2)["done"] == 4
    assert n_done(jobs) == 4


@pytest.mark.slow  # ~10 s: real subprocess, hard kill, rerun
def test_kill_and_resume(ws):
    """Hard-kill the CLI mid-matrix, rerun it: finished runs are skipped, the rest complete, no .tmp left."""
    m = write_matrix(ws, {"A": {"envs": envs(ws, "n1_k06"), "policies": ["ppo"], "seeds": list(range(6))}},
                     builders={"ppo": "fakes:build_slow"})
    cmd = [sys.executable, str(REPO / "scripts" / "run_matrix.py"), str(m), "--runs-root", str(ws / "runs"),
           "--crn-root", str(ws / "crn"), "--configs-root", str(ws / "configs")]
    env = os.environ | {"PYTHONPATH": str(TESTS)}
    jobs = jobs_for(ws, m)

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 60
    while n_done(jobs) < 2 and time.time() < deadline and proc.poll() is None:
        time.sleep(0.1)
    proc.kill()
    proc.wait()
    before = n_done(jobs)
    assert 2 <= before < 6, "the kill should land mid-matrix"

    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert f"skipped/blocked={before}" in out.stdout
    assert n_done(jobs) == 6
    assert not list((ws / "runs").rglob("*.tmp"))
