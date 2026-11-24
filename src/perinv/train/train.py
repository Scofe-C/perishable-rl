"""RL training builder for the runner (WAREHOUSE.md §7): build(ctx) trains or reloads, then returns a Policy.

    policy = train.build(ctx)                       # called by perinv.runner for ppo, a2c, sac, dqn, maskppo
    model, vecnorm, curve = train.train_model(cfg, "ppo", hparams, seed=0, steps=10_000)   # tuning, smoke test

Run folder: model.zip, vecnormalize.pkl (if used), learning_curve.csv. The manifest gets
train = {train_key, status, actual_env_steps, train_seconds, train_cpu_seconds} and actual_env_steps.
A reused model (same train_key, e.g. a cross-evaluation or new test paths) copies those numbers from the run
that trained it, so CPU-hours stay the training cost and are not counted twice.

Owner: Member 2.
"""

from __future__ import annotations

import time
from pathlib import Path

from perinv.agents.factory import check_algo, load_model, make_model, make_train_env, split_hparams
from perinv.agents.sb3_policy import SB3Policy, load_vecnormalize
from perinv.data.crn import load_paths
from perinv.env.config import EnvConfig
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.train.callbacks import CURVE_FILE, LearningCurve
from perinv.utils.run_io import read_manifest

MODEL_FILE = "model.zip"
VECNORM_FILE = "vecnormalize.pkl"
CURVE_POINTS = 20   # learning-curve evaluations per run, unless budget.yaml sets eval_every_env_steps


def train_model(cfg: EnvConfig, algo: str, hparams: dict, *, seed: int, steps: int, make_env=PerishableInventoryEnv,
                tune_demand=None, curve_every: int | None = None, out_dir: Path | None = None):
    """Train one model. Returns (model, vecnormalize or None, learning-curve rows)."""
    check_algo(algo, cfg)
    ours, sb3_kwargs = split_hparams(hparams)
    venv = make_train_env(cfg, algo, seed=seed, make_env=make_env, n_envs=ours["n_envs"],
                          normalize_obs=ours["normalize_obs"], normalize_reward=ours["normalize_reward"],
                          gamma=float(sb3_kwargs.get("gamma", 0.99)))
    vecnorm = venv if hasattr(venv, "obs_rms") else None
    model = make_model(algo, venv, sb3_kwargs, seed)
    callback = None
    if tune_demand is not None:
        callback = LearningCurve(algo=algo, cfg=cfg, make_env=make_env, tune_demand=tune_demand,
                                 every=curve_every or max(steps // CURVE_POINTS, 1),
                                 out=out_dir / CURVE_FILE if out_dir else None, vecnormalize_of=lambda: vecnorm)
    model.learn(total_timesteps=int(steps), callback=callback, progress_bar=False)
    if out_dir is not None:
        model.save(out_dir / MODEL_FILE)
        if vecnorm is not None:
            vecnorm.save(str(out_dir / VECNORM_FILE))
    return model, vecnorm, (callback.rows if callback else [])


def build(ctx) -> SB3Policy:
    algo = ctx.policy
    check_algo(algo, ctx.cfg)
    if ctx.reuse_model:
        model_dir = Path(ctx.model_dir)
        model = load_model(algo, model_dir / MODEL_FILE)
        vecnorm = load_vecnormalize(model_dir / VECNORM_FILE)
        trained = read_manifest(model_dir) or {}
        train = dict(trained.get("train") or {}) | {"reused_from": model_dir.name}
        ctx.run.update(train=train, actual_env_steps=trained.get("actual_env_steps"))
        return SB3Policy(model, algo, ctx.cfg, vecnorm)

    if not ctx.train_env_steps:
        raise ValueError("train_env_steps is not set: fill configs/budget.yaml or run with --profile tiny|smoke")
    tune = load_paths(ctx.cfg, "tune", ctx.crn_root)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    ctx.run.log("train_start", algo=algo, steps=ctx.train_env_steps)
    model, vecnorm, curve = train_model(ctx.cfg, algo, ctx.hparams, seed=ctx.seed, steps=ctx.train_env_steps,
                                        make_env=ctx.make_env, tune_demand=tune, out_dir=ctx.run.dir)
    actual = int(model.num_timesteps)
    train = {"train_key": ctx.run.train_key, "status": "done", "actual_env_steps": actual,
             "train_seconds": round(time.perf_counter() - wall0, 2),
             "train_cpu_seconds": round(time.process_time() - cpu0, 2),
             "final_tune_profit": curve[-1]["tune_profit"] if curve else None}
    ctx.run.update(train=train, actual_env_steps=actual)
    ctx.run.log("train_done", **{k: v for k, v in train.items() if k != "train_key"})
    return SB3Policy(model, algo, ctx.cfg, vecnorm)
