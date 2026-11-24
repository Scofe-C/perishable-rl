"""TEST DOUBLE, not the env. A minimal SPEC §3 implementation so eval/runner tests can run before M1's env exists.

Do not import this outside tests. When `perinv.env.perishable_env` lands, the eval tests should also run against
it (same reset/step/info contract), and this file only stays for fast unit tests.
"""

from __future__ import annotations

import numpy as np

from perinv.env.config import EnvConfig


def tiny_config(n_skus: int = 2, capacity: int = 20, horizon: int = 7) -> EnvConfig:
    return EnvConfig(
        n_skus=n_skus, capacity=capacity, shelf_life=3, lead_time=2, horizon=horizon,
        demand_mean=(4.0,) * n_skus, price=(10.0,) * n_skus, order_cost=(4.0,) * n_skus,
        holding_cost=(0.1,) * n_skus, expiry_cost=(2.0,) * n_skus, lost_sales_cost=(2.0,) * n_skus,
        name="tiny",
    )


class FakeEnv:
    """reset(seed, options={'demand_path': int[T, N]}) / step(q) with the SPEC §6 info dict."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.n, self.L, self.P = cfg.n_skus, cfg.shelf_life, cfg.lead_time

    def _state(self) -> dict:
        return {"inv": self.inv.copy(), "pipe": self.pipe.copy(), "t": self.t}

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.inv.ravel(), self.pipe.ravel(), [self.t]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.demand = np.asarray(options["demand_path"])
        from perinv.env.dynamics import initial_state  # the SPEC §3 start; only the day itself is re-implemented here

        self.inv, self.pipe = initial_state(self.cfg)
        self.t = 0
        return self._obs(), {"state": self._state()}

    def step(self, action):
        c = self.cfg
        q = np.maximum(np.asarray(action, np.int64).reshape(self.n), 0).copy()
        occ = int(self.inv.sum() + self.pipe.sum())
        remaining = c.capacity - occ
        clipped = bool(q.sum() > remaining)
        while q.sum() > remaining:            # 1. clip, largest orders first
            q[np.argmax(q)] -= 1
        self.inv[:, -1] += self.pipe[:, 0]    # 2. arrive
        d = self.demand[self.t].astype(np.int64)
        left = d.copy()
        for k in range(self.L):               # 3. FIFO from the oldest bucket
            take = np.minimum(self.inv[:, k], left)
            self.inv[:, k] -= take
            left -= take
        sold, lost = d - left, left
        expired = self.inv[:, 0].copy()       # 4. expire + age
        self.inv[:, :-1] = self.inv[:, 1:]
        self.inv[:, -1] = 0
        held = self.inv.sum(axis=1)           # 5. hold
        self.pipe[:, :-1] = self.pipe[:, 1:]  # 6. pipeline
        self.pipe[:, -1] = q
        p, oc, h, w, b = (np.asarray(getattr(c, f)) for f in
                          ("price", "order_cost", "holding_cost", "expiry_cost", "lost_sales_cost"))
        reward = float((p * sold - oc * q - h * held - w * expired - b * lost).sum())  # 7. reward
        self.t += 1
        info = {"sold": sold, "demand": d, "lost": lost, "expired": expired, "ordered": q, "held": held,
                "occupancy": occ, "clipped": clipped, "state": self._state()}
        return self._obs(), reward, self.t == c.horizon, False, info   # D8: terminated at T


class ConstantPolicy:
    def __init__(self, q):
        self.q = np.asarray(q, np.int64)

    def reset(self) -> None:
        pass

    def act(self, obs, info):
        return self.q.copy()


class RandomPolicy:
    """Draws from the global numpy RNG, which evaluate() seeds with the path index."""

    def __init__(self, n_skus: int, high: int = 8):
        self.n, self.high = n_skus, high

    def reset(self) -> None:
        pass

    def act(self, obs, info):
        return np.random.randint(0, self.high, size=self.n).astype(np.int64)


# ---------------------------------------------------------------- runner builders ("fakes:build_*")
# Builders get a perinv.runner.BuildContext and return a Policy. These stand in for M2's train.build.

def _record_build(ctx):
    """Pretends to train: records 'train' or 'reuse:<run that holds the model>', then marks training done."""
    with open(ctx.run.dir / "builds.txt", "a") as fh:
        fh.write((f"reuse:{ctx.model_dir.name}" if ctx.reuse_model else "train") + "\n")
    ctx.run.update(train={"train_key": ctx.run.train_key, "status": "done"}, actual_env_steps=ctx.train_env_steps)


def build_constant(ctx):
    _record_build(ctx)
    return ConstantPolicy([2] * ctx.cfg.n_skus)


def build_belief(ctx):
    """Orders round(lambda) per SKU every day, lambda taken from the BUILD config: the policy's belief."""
    _record_build(ctx)
    return ConstantPolicy([round(x) for x in ctx.cfg.demand_mean])


def build_random(ctx):
    return RandomPolicy(ctx.cfg.n_skus, high=6)


def build_crash(ctx):
    raise FloatingPointError("NaN in policy loss")


def build_slow(ctx):
    import time

    time.sleep(1.0)
    return ConstantPolicy([1] * ctx.cfg.n_skus)


# ---------------------------------------------------------------- fault injection (tests/test_faults.py)

class FixedActionPolicy:
    """Returns exactly the given object every day, so tests can feed evaluate() malformed actions."""

    def __init__(self, action):
        self.action = action

    def reset(self) -> None:
        pass

    def act(self, obs, info):
        return self.action


BAD_ACTIONS = {
    "nan": lambda n: np.full(n, np.nan),
    "negative": lambda n: np.full(n, -1),
    "fraction": lambda n: np.full(n, 1.5),
    "wrong_shape": lambda n: np.ones(n + 1, np.int64),
    "text": lambda n: np.array(["2"] * n),
}


def build_bad_action(ctx):
    """Policy name 'bad_<kind>' selects the malformed action from BAD_ACTIONS."""
    return FixedActionPolicy(BAD_ACTIONS[ctx.policy.removeprefix("bad_")](ctx.cfg.n_skus))


def build_scalar(ctx):
    """A Discrete-style 0-d action: valid for N = 1."""
    return FixedActionPolicy(np.int64(2))


def build_die(ctx):
    """Hard crash after the run started: no exception, no cleanup (like a segfault or an OOM kill)."""
    import os

    os._exit(3)


class NanRewardEnv(FakeEnv):
    """A buggy env: the reward of day 1 is NaN."""

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        return obs, (float("nan") if self.t == 2 else reward), term, trunc, info
