"""Hyperparameter tuning (EXPERIMENTS.md rule 2): Optuna TPE, fixed seed, exactly n_trials per algorithm × cell,
scored on the **tune** CRN paths only.

    python -m perinv.train.tune --algos ppo,a2c --envs configs/env/n1_k06.yaml,configs/env/n1_k10.yaml --jobs 4
    python -m perinv.train.tune --algos sac --envs configs/env/n5_k10.yaml --steps 50000 --trials 10

Each trial trains one model (seed 0) with the search-space values from configs/tune/<algo>.yaml on top of
configs/algo/<algo>.yaml, then scores mean profit on the tune paths. Trial 0 is always the untuned defaults, so
tuning never ends worse than the defaults on the tune paths. The winner is written to
configs/tuned/<algo>-<env stem>.yaml, which the runner prefers over the defaults (runner.load_hparams), and
every trial is listed in results/tune/<algo>-<env stem>.csv. The step budget defaults to budget.yaml's
train_env_steps, so tuning uses the same budget as the final runs.

Owner: Member 2.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import optuna
import pandas as pd
import yaml

from perinv.agents.sb3_policy import SB3Policy
from perinv.data.crn import load_paths
from perinv.env.config import load_env_config
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.eval.evaluate import evaluate
from perinv.train.train import train_model
from perinv.utils.seeding import init_process

TUNE_SEED = 0
log = logging.getLogger("perinv.tune")


def suggest(trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for name, spec in (space or {}).items():
        kind = spec["type"]
        if kind == "float":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        elif kind == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown search-space type {kind!r} for {name}")
    return out


def tune_one(algo: str, env_path: str, *, steps: int, n_trials: int, configs_root: Path, crn_root: str,
             out_root: Path) -> dict:
    """Run exactly n_trials for one algorithm × cell; write the tuned YAML and the trial table."""
    init_process(TUNE_SEED)
    cfg = load_env_config(env_path)
    defaults = (yaml.safe_load((configs_root / "algo" / f"{algo}.yaml").read_text()) or {}).get("hparams") or {}
    space = yaml.safe_load((configs_root / "tune" / f"{algo}.yaml").read_text()) or {}
    tune_demand = load_paths(cfg, "tune", crn_root)
    env = PerishableInventoryEnv(cfg)

    def objective(trial: optuna.Trial) -> float:
        hp = dict(defaults) if trial.number == 0 else defaults | suggest(trial, space)   # trial 0: defaults
        t0 = time.perf_counter()
        model, vecnorm, _ = train_model(cfg, algo, hp, seed=TUNE_SEED, steps=steps)
        profit = float(evaluate(SB3Policy(model, algo, cfg, vecnorm), env, tune_demand).per_path()["profit"].mean())
        trial.set_user_attr("seconds", round(time.perf_counter() - t0, 1))
        trial.set_user_attr("actual_env_steps", int(model.num_timesteps))
        return profit

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    stem = Path(env_path).stem
    best = defaults | study.best_trial.params
    tuned = configs_root / "tuned" / f"{algo}-{stem}.yaml"
    tuned.parent.mkdir(parents=True, exist_ok=True)
    tuned.write_text(yaml.safe_dump({
        "algo": algo, "env": env_path, "hparams": best,
        "x-tuning": {"tune_profit": study.best_value, "default_tune_profit": study.trials[0].value,
                     "n_trials": n_trials, "train_env_steps": steps, "seed": TUNE_SEED,
                     "scored_on": "tune CRN paths", "best_trial": study.best_trial.number}},
        sort_keys=False), encoding="utf-8")
    out_root.mkdir(parents=True, exist_ok=True)
    table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    table.to_csv(out_root / f"{algo}-{stem}.csv", index=False, lineterminator="\n")
    summary = {"algo": algo, "env": stem, "best": study.best_value, "default": study.trials[0].value,
               "best_trial": study.best_trial.number, "file": tuned.as_posix()}
    log.info("tuned %s on %s: %.2f (defaults %.2f) -> %s", algo, stem, summary["best"], summary["default"], tuned)
    return summary


def _worker(args: tuple) -> dict:
    algo, env_path, kw = args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        return tune_one(algo, env_path, **kw)
    except Exception as exc:   # one failed cell must not stop the others
        return {"algo": algo, "env": Path(env_path).stem, "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", required=True, help="comma list, e.g. ppo,a2c,sac,dqn,maskppo")
    ap.add_argument("--envs", required=True, help="comma list of env YAMLs")
    ap.add_argument("--steps", type=int, help="env steps per trial (default: budget.yaml train_env_steps)")
    ap.add_argument("--trials", type=int, help="default: budget.yaml n_tune_trials")
    ap.add_argument("--jobs", type=int, default=1, help="algorithm × cell pairs tuned in parallel")
    ap.add_argument("--configs-root", default="configs")
    ap.add_argument("--crn-root", default="data/crn")
    ap.add_argument("--out", default="results/tune")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    configs_root = Path(args.configs_root)
    budget = yaml.safe_load((configs_root / "budget.yaml").read_text())
    steps = args.steps or budget.get("train_env_steps")
    trials = args.trials or budget["n_tune_trials"]
    if not steps:
        ap.error("no step budget: pass --steps or set train_env_steps in configs/budget.yaml")
    kw = {"steps": int(steps), "n_trials": int(trials), "configs_root": configs_root, "crn_root": args.crn_root,
          "out_root": Path(args.out)}
    work = [(a, e, kw) for a in args.algos.split(",") for e in args.envs.split(",")
            if not (a in ("dqn", "maskppo") and load_env_config(e).n_skus != 1)]
    if args.jobs > 1:
        with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
            results = pool.map(_worker, work, chunksize=1)
    else:
        results = [_worker(w) for w in work]
    print(pd.DataFrame(results).to_string(index=False))
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
