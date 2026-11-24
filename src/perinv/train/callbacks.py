"""Training callbacks: learning curve on the tune paths, against actual env steps (EXPERIMENTS.md metrics).

SB3's total_timesteps is only a lower bound (on-policy methods finish their rollout), so the curve and the
manifest use model.num_timesteps, the steps actually taken. The curve is scored with the same evaluate() and a
frozen copy of the VecNormalize statistics at that moment; it never reads the test paths.

Owner: Member 2.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from perinv.agents.sb3_policy import SB3Policy
from perinv.env.config import EnvConfig
from perinv.eval.evaluate import evaluate

CURVE_FILE = "learning_curve.csv"
CURVE_FIELDS = ["env_steps", "tune_profit", "wall_seconds"]


class LearningCurve(BaseCallback):
    def __init__(self, *, algo: str, cfg: EnvConfig, make_env, tune_demand: np.ndarray, every: int,
                 out: Path | None, vecnormalize_of=lambda: None):
        super().__init__()
        self.algo, self.cfg, self.tune, self.every, self.out = algo, cfg, tune_demand, max(int(every), 1), out
        self.env = make_env(cfg)
        self.vecnormalize_of = vecnormalize_of
        self.next_at = 0
        self.rows: list[dict] = []
        self.t0 = time.perf_counter()

    def score(self) -> float:
        policy = SB3Policy(self.model, self.algo, self.cfg, self.vecnormalize_of())
        return float(evaluate(policy, self.env, self.tune).per_path()["profit"].mean())

    def record(self) -> None:
        self.rows.append({"env_steps": int(self.num_timesteps), "tune_profit": self.score(),
                          "wall_seconds": round(time.perf_counter() - self.t0, 2)})
        if self.out is not None:
            with open(self.out, "w", newline="\n", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, CURVE_FIELDS, lineterminator="\n")
                w.writeheader()
                w.writerows(self.rows)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.record()
            self.next_at = self.num_timesteps + self.every
        return True

    def _on_training_end(self) -> None:
        if not self.rows or self.rows[-1]["env_steps"] != self.num_timesteps:
            self.record()
