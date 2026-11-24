"""Benchmark env and training throughput to choose the VecEnv and set train_env_steps (configs/budget.yaml).

    python scripts/bench_vecenv.py                 # about 1–2 minutes
    python scripts/bench_vecenv.py --ppo-steps 0   # env-only numbers

Measures (one CPU thread per process, as in real runs):
1. raw env steps/s (N = 1 and N = 5, through CapacityShareAction);
2. DummyVecEnv vs SubprocVecEnv (spawn) steps/s with random actions, n_envs in {1, 4, 8};
3. PPO learn() env steps/s with DummyVecEnv and SubprocVecEnv, n_envs in {1, 4}.
Run it on the machine that runs the final experiments (Linux/WSL2): Windows spawn costs differ.

Owner: all (week 1). Contract: docs/EXPERIMENTS.md fairness rule 1.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

from perinv.env.config import load_env_config
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.env.wrappers import CapacityShareAction
from perinv.utils.seeding import init_process


def make(path: str):
    return lambda: CapacityShareAction(PerishableInventoryEnv(load_env_config(path)))


def raw_steps_per_second(path: str, steps: int) -> float:
    env = make(path)()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (steps, env.action_space.shape[0])).astype(np.float32)
    t0 = time.perf_counter()
    for a in actions:
        _, _, _, truncated, _ = env.step(a)
        if truncated:
            env.reset()
    return steps / (time.perf_counter() - t0)


def vec_steps_per_second(path: str, kind: str, n_envs: int, steps: int) -> float:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec = DummyVecEnv([make(path)] * n_envs) if kind == "dummy" else SubprocVecEnv([make(path)] * n_envs, "spawn")
    try:
        vec.reset()
        rng = np.random.default_rng(0)
        n_calls = max(1, steps // n_envs)
        actions = rng.uniform(-1, 1, (n_calls, n_envs, vec.action_space.shape[0])).astype(np.float32)
        t0 = time.perf_counter()
        for a in actions:
            vec.step(a)
        return n_calls * n_envs / (time.perf_counter() - t0)
    finally:
        vec.close()


def ppo_steps_per_second(path: str, kind: str, n_envs: int, steps: int) -> float:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec = DummyVecEnv([make(path)] * n_envs) if kind == "dummy" else SubprocVecEnv([make(path)] * n_envs, "spawn")
    try:
        model = PPO("MlpPolicy", vec, n_steps=max(64, 2048 // n_envs), batch_size=64, seed=0, device="cpu", verbose=0)
        model.learn(256)   # warm-up: first-call overheads
        t0 = time.perf_counter()
        model.learn(steps, reset_num_timesteps=False)
        return steps / (time.perf_counter() - t0)
    finally:
        vec.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-steps", type=int, default=20000)
    ap.add_argument("--ppo-steps", type=int, default=8192)
    args = ap.parse_args()
    print(f"# {platform.platform()} · Python {platform.python_version()}")
    for path in ("configs/env/n1_k06.yaml", "configs/env/n5_k06.yaml"):
        print(f"\n## {path}")
        print(f"raw env (CapacityShareAction): {raw_steps_per_second(path, args.env_steps):,.0f} steps/s")
        for kind in ("dummy", "subproc"):
            for n in (1, 4, 8):
                print(f"{kind:<8} n_envs={n}: {vec_steps_per_second(path, kind, n, args.env_steps):>9,.0f} steps/s")
        if args.ppo_steps:
            for kind in ("dummy", "subproc"):
                for n in (1, 4):
                    rate = ppo_steps_per_second(path, kind, n, args.ppo_steps)
                    print(f"PPO {kind:<8} n_envs={n}: {rate:>7,.0f} env steps/s "
                          f"(1M steps ≈ {1e6 / rate / 60:.0f} min)")
    return 0


if __name__ == "__main__":
    init_process(0)
    sys.exit(main())
