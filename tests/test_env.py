"""PerishableInventoryEnv and the action wrappers (SPEC §3–§6): the gym contract, exact CRN replay, capacity,
the D1 mapping endpoints, masks through SB3 wrappers, and agreement with the test double inside evaluate()."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import yaml
from fakes import ConstantPolicy, FakeEnv, RandomPolicy, tiny_config
from gymnasium.utils.env_checker import check_env
from hypothesis import given, settings
from hypothesis import strategies as st
from test_runner import ENV_BASE

from perinv.data.crn import generate_paths, write_paths
from perinv.env.config import load_env_config
from perinv.env.perishable_env import PerishableInventoryEnv
from perinv.env.wrappers import (
    CapacityShareAction,
    DiscreteOrder,
    capacity_share_orders,
    largest_remainder,
    total_share,
)
from perinv.eval import warehouse
from perinv.eval.evaluate import evaluate
from perinv.runner import expand, run_matrix

CFG1 = tiny_config(n_skus=1, capacity=12, horizon=10)
CFG3 = tiny_config(n_skus=3, capacity=30, horizon=10)
SPEC6_KEYS = {"sold", "demand", "lost", "expired", "ordered", "held", "occupancy", "clipped", "reward_breakdown",
              "state"}


# ---------------------------------------------------------------- gym contract

@pytest.mark.parametrize("make", [
    lambda: PerishableInventoryEnv(CFG1),
    lambda: PerishableInventoryEnv(CFG3),
    lambda: CapacityShareAction(PerishableInventoryEnv(CFG3)),
    lambda: DiscreteOrder(PerishableInventoryEnv(CFG1)),
])
@pytest.mark.filterwarnings("ignore:.*different from the unwrapped version")   # wrapped on purpose
def test_gymnasium_check_env(make):
    check_env(make(), skip_render_check=True)


def test_sb3_check_env_on_the_box_wrapper():
    from stable_baselines3.common.env_checker import check_env as sb3_check

    sb3_check(CapacityShareAction(PerishableInventoryEnv(CFG3)), warn=True)


def test_episode_shape_and_info():
    env = PerishableInventoryEnv(CFG3)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and set(info) == {"state"}
    for t in range(CFG3.horizon):
        obs, r, terminated, truncated, info = env.step(np.array([1, 2, 3]))
        assert env.observation_space.contains(obs) and set(info) == SPEC6_KEYS
        assert terminated is False and truncated is (t == CFG3.horizon - 1)
        b = info["reward_breakdown"]
        assert r == pytest.approx(b["revenue"] - b["ordering"] - b["holding"] - b["expiry"] - b["lost_sales"])
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0, 0, 0]))


def test_reset_starts_from_the_warm_state():
    """D4: on hand empty, pipeline pre-filled with round(lambda * season) (CFG1: lambda 4, tau 2 -> [4, 4])."""
    env = PerishableInventoryEnv(CFG1)
    _, info = env.reset(seed=0)
    assert info["state"]["inv"].tolist() == [[0, 0, 0]] and info["state"]["pipe"].tolist() == [[4, 4]]


def test_occupancy_is_decision_time_and_obs_matches_spec5():
    env = PerishableInventoryEnv(CFG1)
    env.reset(options={"demand_path": np.zeros((10, 1), np.int64)})
    before = env.occupancy
    _, _, _, _, info = env.step(np.array([3]))
    assert info["occupancy"] == before == 8             # D3: before today's order (warm pipeline holds 8)
    obs, _, _, _, info = env.step(np.array([0]))
    assert info["occupancy"] == before + 3
    L, P, C = CFG1.shelf_life, CFG1.lead_time, CFG1.capacity
    inv, pipe = info["state"]["inv"], info["state"]["pipe"]
    expected = np.concatenate([inv.ravel() / C, pipe.ravel() / C, np.eye(7)[2], [2 / 10, (inv.sum() + pipe.sum()) / C]])
    assert obs.shape == (L + P + 9,) and np.allclose(obs, expected)


# ---------------------------------------------------------------- demand: CRN replay and training draws

def test_crn_path_replays_exactly_regardless_of_seed():
    path = generate_paths(CFG3, "test", 1, 7)[0]
    runs = []
    for seed in (0, 1):
        env = PerishableInventoryEnv(CFG3)
        env.reset(seed=seed, options={"demand_path": path})
        runs.append([env.step(np.array([2, 2, 2]))[1:] for _ in range(CFG3.horizon)])
    assert [r for r, *_ in runs[0]] == [r for r, *_ in runs[1]]
    assert np.array_equal(np.stack([i["demand"] for *_, i in runs[0]]), path)


def test_training_demand_is_seeded_and_poisson():
    cfg = replace(CFG1, horizon=70, season=(0.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0))
    def demand(seed):
        env = PerishableInventoryEnv(cfg)
        env.reset(seed=seed)
        return np.array([env.step(np.array([0]))[-1]["demand"][0] for _ in range(70)])
    assert np.array_equal(demand(3), demand(3)) and not np.array_equal(demand(3), demand(4))
    draws = np.stack([demand(s) for s in range(300)])          # [seed, day]
    assert draws[:, 0::7].mean() == pytest.approx(2.0, rel=0.1)   # 4 * 0.5 on Mondays
    assert draws[:, 1::7].mean() == pytest.approx(6.0, rel=0.1)   # 4 * 1.5 on Tuesdays


@pytest.mark.parametrize("path, match", [(np.zeros((9, 1), np.int64), "shape"), (np.full((10, 1), 1.5), "integers"),
                                         (np.full((10, 1), -1), "integers")])
def test_bad_demand_path_is_rejected(path, match):
    with pytest.raises(ValueError, match=match):
        PerishableInventoryEnv(CFG1).reset(options={"demand_path": path})


# ---------------------------------------------------------------- capacity and masks

@settings(max_examples=40, deadline=None)
@given(st.integers(1, 3), st.integers(1, 25), st.integers(0, 2**32 - 1))
def test_capacity_never_exceeded(n, cap, seed):
    cfg = tiny_config(n_skus=n, capacity=cap, horizon=15)
    env = PerishableInventoryEnv(cfg)
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    for _ in range(15):
        occ = env.occupancy
        _, _, _, _, info = env.step(rng.integers(0, 2 * cap + 2, n))     # often infeasible: the env clips
        assert occ + info["ordered"].sum() <= cap and env.occupancy <= cap


def test_action_masks_follow_remaining_capacity():
    env = PerishableInventoryEnv(CFG1)
    env.reset(options={"demand_path": np.zeros((10, 1), np.int64)})
    assert env.action_masks().tolist() == [k <= 4 for k in range(13)]        # C 12 - warm pipeline 8
    env.step(np.array([2]))
    assert env.action_masks().tolist() == [k <= 2 for k in range(13)]
    assert env.action_masks().tolist() == (np.arange(13) <= env.remaining).tolist()
    with pytest.raises(ValueError, match="N = 1"):
        PerishableInventoryEnv(CFG3).action_masks()


def _masked_env(cfg):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(DiscreteOrder(PerishableInventoryEnv(cfg)))


def test_maskable_ppo_gets_masks_through_wrappers_and_never_clips():
    """Pitfall guard: masks reach MaskablePPO through Monitor + DiscreteOrder (gymnasium 1.x get_wrapper_attr),
    and a masked policy never needs clipping."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: _masked_env(CFG1)] * 2)
    vec.reset()
    assert get_action_masks(vec).shape == (2, 13)
    model = MaskablePPO("MlpPolicy", vec, n_steps=16, batch_size=16, n_epochs=1, seed=0, device="cpu")
    model.learn(32)
    env = _masked_env(CFG1)
    obs, _ = env.reset(seed=1)
    clipped = 0
    for _ in range(CFG1.horizon):
        action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=False)
        obs, _, _, _, info = env.step(action)
        clipped += info["clipped"]
    assert clipped == 0


@pytest.mark.slow  # spawns worker processes
def test_masks_reach_subproc_workers():
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv

    cfg = CFG1
    vec = SubprocVecEnv([lambda: _masked_env(cfg)] * 2, start_method="spawn")
    try:
        vec.reset()
        vec.step(np.array([2, 0]))
        masks = get_action_masks(vec)
        remaining = vec.get_attr("remaining")          # per worker; demand is random in training mode
        for i in range(2):
            assert masks[i].tolist() == (np.arange(13) <= remaining[i]).tolist()
    finally:
        vec.close()


# ---------------------------------------------------------------- Box mapping (D1) and the discrete wrapper

def test_share_endpoints_are_exact_for_every_R():
    assert total_share(-1.0) == 0.0 and total_share(1.0) == 1.0 and total_share(0.0) == pytest.approx(0.5)
    for R in range(0, 1001):
        assert capacity_share_orders(np.array([-1.0, 0.0]), R).sum() == 0
        assert capacity_share_orders(np.array([1.0, 0.0]), R).sum() == R


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(-3, 3), min_size=2, max_size=6), st.integers(0, 200))
def test_mapping_is_feasible_integer_and_exact_total(action, R):
    q = capacity_share_orders(np.array(action), R)
    total = int(np.clip(np.floor(total_share(action[0]) * R + 0.5), 0, R))
    assert q.dtype == np.int64 and (q >= 0).all() and q.sum() == total <= R


def test_even_split_gives_remainders_to_the_lowest_index():
    assert capacity_share_orders(np.array([1.0, 0.3, 0.3, 0.3]), 10).tolist() == [4, 3, 3]
    assert largest_remainder(np.array([0.5, 0.5]), 3).tolist() == [2, 1]


def test_larger_logit_gets_more():
    q = capacity_share_orders(np.array([1.0, 1.0, -1.0]), 20)
    assert q[0] > q[1] and q.sum() == 20


@pytest.mark.parametrize("n", [2, 5])
def test_one_sku_can_take_all_capacity(n):
    """D7: the split must not cap a single SKU (plain softmax on [-1, 1] capped it at 65 % for N = 5)."""
    a = np.array([1.0, 1.0] + [-1.0] * (n - 1))
    assert capacity_share_orders(a, 60).tolist() == [60] + [0] * (n - 1)


def test_box_wrapper_reads_remaining_capacity():
    env = CapacityShareAction(PerishableInventoryEnv(CFG3))
    env.reset(options={"demand_path": np.zeros((10, 3), np.int64)})
    assert env.unwrapped.remaining == 6                                        # 30 - warm pipeline 3 x 2 x 4
    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0], np.float32))    # fill everything
    assert info["ordered"].tolist() == [2, 2, 2] and not info["clipped"]
    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0], np.float32))    # nothing left
    assert info["ordered"].sum() == 0


def test_discrete_wrapper():
    env = DiscreteOrder(PerishableInventoryEnv(CFG1))
    assert env.action_space == gym.spaces.Discrete(13)
    env.reset(seed=0)
    assert env.step(4)[-1]["ordered"].tolist() == [4]
    with pytest.raises(ValueError, match="N = 1"):
        DiscreteOrder(PerishableInventoryEnv(CFG3))


# ---------------------------------------------------------------- inside the pipeline

@pytest.mark.parametrize("cfg, policy", [(CFG1, lambda: ConstantPolicy([5])), (CFG3, lambda: RandomPolicy(3, 15))])
def test_evaluate_gives_identical_results_on_env_and_test_double(cfg, policy):
    demand = generate_paths(cfg, "test", 8, 3)
    real, fake = evaluate(policy(), PerishableInventoryEnv(cfg), demand), evaluate(policy(), FakeEnv(cfg), demand)
    # quantities identical; rewards equal up to float summation order (~1e-14)
    assert np.allclose(real.day_reward, fake.day_reward, rtol=0, atol=1e-9)
    assert np.array_equal(real.clipped, fake.clipped)
    assert all(np.array_equal(real.qty[f], fake.qty[f]) for f in real.qty)


def test_runner_end_to_end_with_the_real_env(tmp_path):
    (tmp_path / "env").mkdir()
    p = tmp_path / "env" / "e.yaml"
    p.write_text(yaml.safe_dump(ENV_BASE | {"n_skus": 2, "kappa": 1.0, "capacity": 20}))
    write_paths(load_env_config(p), tmp_path / "crn", root_seed=1, sizes={"tune": 2, "test": 5})
    (tmp_path / "budget.yaml").write_text(yaml.safe_dump({"train_env_steps": 10, "seeds": [0], "profiles": {}}))
    m = tmp_path / "m.yaml"
    m.write_text(yaml.safe_dump({"budget": (tmp_path / "budget.yaml").as_posix(),
                                 "builders": {"ppo": "fakes:build_random"},       # default env factory = real env
                                 "groups": {"G": {"envs": [p.as_posix()], "policies": ["ppo"]}}}))
    jobs = expand(m, runs_root=(tmp_path / "runs").as_posix(), crn_root=(tmp_path / "crn").as_posix(),
                  configs_root=tmp_path)
    assert jobs[0].env_factory == "perinv.env.perishable_env:PerishableInventoryEnv"
    assert run_matrix(jobs)["done"] == 1
    assert len(warehouse.connect(tmp_path / "runs").sql("SELECT * FROM v_path").df()) == 5
