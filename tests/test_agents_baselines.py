"""Guard tests for the team draft §7 pitfalls owned by M2 (they replaced the skipped placeholders)."""

import numpy as np
import pytest
import yaml

from perinv.agents.sb3_policy import SB3Policy
from perinv.baselines import newsvendor, s_S
from perinv.baselines.dp_policy import DPPolicy, load_or_solve
from perinv.data.crn import load_paths
from perinv.env.config import load_env_config
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.eval.evaluate import evaluate
from perinv.train.train import train_model

CFG = load_env_config("configs/env/n1_k06.yaml")
CFG5 = load_env_config("configs/env/n5_k10.yaml")


def per_path(policy, cfg, demand, order=None):
    r = evaluate(policy, PerishableInventoryEnv(cfg), demand, path_ids=order)
    return r, r.per_path().set_index("path_id")["profit"]


def test_sb3_adapter_frozen_vecnormalize_and_deterministic():
    """VecNormalize must not update at evaluation: shuffling the path order changes nothing, and the training
    statistics are not touched by evaluation."""
    hp = {"n_envs": 2, "n_steps": 64, "batch_size": 64, "normalize_obs": True, "normalize_reward": True,
          "net_arch": [16, 16]}
    model, vecnorm, _ = train_model(CFG, "ppo", hp, seed=0, steps=256)
    policy = SB3Policy(model, "ppo", CFG, vecnorm)
    assert policy.vecnormalize.training is False and policy.vecnormalize.norm_reward is False
    mean_before = vecnorm.obs_rms.mean.copy()
    demand = load_paths(CFG, "test")[:8]
    _, a = per_path(policy, CFG, demand)
    _, b = per_path(policy, CFG, demand, order=[7, 3, 0, 5, 1, 6, 2, 4])
    assert a.sort_index().equals(b.sort_index())
    np.testing.assert_array_equal(vecnorm.obs_rms.mean, mean_before)
    np.testing.assert_array_equal(policy.vecnormalize.obs_rms.mean, mean_before)


@pytest.mark.parametrize("algo", ["sac", "maskppo"])
def test_sb3_adapter_returns_feasible_order_quantities(algo):
    """Box: the adapter maps the raw action through capacity_share_orders; MaskablePPO: masks at evaluation.
    Neither may ever be clipped by the env."""
    cfg = CFG5 if algo == "sac" else CFG
    hp = yaml.safe_load(open(f"configs/algo/{algo}.yaml"))["hparams"] | {"net_arch": [16, 16]}
    if algo == "sac":
        hp |= {"learning_starts": 50, "batch_size": 32}
    else:
        hp |= {"n_envs": 1, "n_steps": 64, "batch_size": 64}
    model, vecnorm, _ = train_model(cfg, algo, hp, seed=0, steps=128)
    r, _ = per_path(SB3Policy(model, algo, cfg, vecnorm), cfg, load_paths(cfg, "test")[:5])
    assert r.clipped.sum() == 0
    assert (r.qty["order"] >= 0).all()


@pytest.mark.parametrize("cfg", [CFG, CFG5], ids=["N1", "N5"])
def test_baselines_feasible_and_deterministic(cfg):
    demand = load_paths(cfg, "test")[:10]
    (s, S), _, _ = s_S.search(cfg, PerishableInventoryEnv, load_paths(cfg, "tune")[:5])
    for policy in (s_S.SSPolicy(cfg, *s_S.per_sku(cfg, s, S)), newsvendor.NewsvendorPolicy(cfg)):
        r1, p1 = per_path(policy, cfg, demand)
        _, p2 = per_path(policy, cfg, demand)
        assert r1.clipped.sum() == 0 and p1.equals(p2)


def test_heuristics_do_not_beat_dp_on_n1():
    demand = load_paths(CFG, "test")
    _, dp = per_path(DPPolicy(load_or_solve(CFG)[0]), CFG, demand)
    _, nv = per_path(newsvendor.NewsvendorPolicy(CFG), CFG, demand)
    (s, S), _, _ = s_S.search(CFG, PerishableInventoryEnv, load_paths(CFG, "tune"))
    _, ss = per_path(s_S.SSPolicy(CFG, *s_S.per_sku(CFG, s, S)), CFG, demand)
    assert nv.mean() <= dp.mean() and ss.mean() <= dp.mean()
