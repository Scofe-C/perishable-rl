"""SB3 / sb3-contrib models and their training envs. No algorithm rewrites (team draft §4).

    venv = make_train_env(cfg, "ppo", seed=0, make_env=PerishableInventoryEnv, n_envs=4, norm_reward=True)
    model = make_model("ppo", venv, sb3_kwargs, seed=0)

Action spaces (SPEC §4): PPO, A2C and SAC use the shared Box mapping (CapacityShareAction). DQN and MaskablePPO
use Discrete(C+1) (DiscreteOrder), N = 1 only. MaskablePPO reads the base env's action_masks() through the
wrappers and the VecEnv.

Our own hparams (popped before the SB3 constructor sees them):
    n_envs          parallel envs in one DummyVecEnv (DECISIONS: DummyVecEnv, not Subproc)
    normalize_obs   VecNormalize on observations (default False: obs are already scaled to [0, 1], SPEC §5)
    normalize_reward VecNormalize on rewards (default True; daily rewards are tens of units)
    net_arch        hidden layers, e.g. [64, 64]; becomes policy_kwargs.net_arch
Everything else goes to the SB3 constructor unchanged.

Owner: Member 2 (PPO, A2C) and Member 3 (SAC, DQN, MaskablePPO).
"""

from __future__ import annotations

from collections.abc import Callable

from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from perinv.env.config import EnvConfig
from perinv.env.wrappers import CapacityShareAction, DiscreteOrder

ALGOS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "dqn": DQN, "maskppo": MaskablePPO}
DISCRETE = {"dqn", "maskppo"}
OUR_KEYS = ("n_envs", "normalize_obs", "normalize_reward", "net_arch")


def split_hparams(hparams: dict) -> tuple[dict, dict]:
    """(our settings with defaults, SB3 constructor kwargs)."""
    hp = dict(hparams or {})
    ours = {"n_envs": int(hp.pop("n_envs", 1)), "normalize_obs": bool(hp.pop("normalize_obs", False)),
            "normalize_reward": bool(hp.pop("normalize_reward", True)), "net_arch": hp.pop("net_arch", None)}
    if ours["net_arch"] is not None:
        hp["policy_kwargs"] = {**hp.get("policy_kwargs", {}), "net_arch": list(ours["net_arch"])}
    return ours, hp


def check_algo(algo: str, cfg: EnvConfig) -> None:
    if algo not in ALGOS:
        raise ValueError(f"unknown algorithm {algo!r}; known: {sorted(ALGOS)}")
    if algo in DISCRETE and cfg.n_skus != 1:
        raise ValueError(f"{algo} uses Discrete(C+1) actions and is N = 1 only (SPEC §4.2), got N = {cfg.n_skus}")


def wrap_actions(env, algo: str):
    return DiscreteOrder(env) if algo in DISCRETE else CapacityShareAction(env)


def make_train_env(cfg: EnvConfig, algo: str, *, seed: int, make_env: Callable[[EnvConfig], object],
                   n_envs: int = 1, normalize_obs: bool = False, normalize_reward: bool = True,
                   gamma: float = 0.99) -> VecEnv:
    """Training envs draw Poisson demand from their own seeded generators (SPEC D5); env i gets seed + i."""
    check_algo(algo, cfg)
    venv = DummyVecEnv([lambda: Monitor(wrap_actions(make_env(cfg), algo)) for _ in range(n_envs)])
    venv.seed(seed)
    if normalize_obs or normalize_reward:
        venv = VecNormalize(venv, norm_obs=normalize_obs, norm_reward=normalize_reward, gamma=gamma)
    return venv


def make_model(algo: str, venv: VecEnv, sb3_kwargs: dict, seed: int):
    kwargs = {"policy": "MlpPolicy", **sb3_kwargs}
    return ALGOS[algo](env=venv, seed=seed, device="cpu", verbose=0, **kwargs)


def load_model(algo: str, path):
    return ALGOS[algo].load(path, device="cpu")
