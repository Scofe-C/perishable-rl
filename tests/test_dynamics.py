"""perinv.env.dynamics: SPEC §3 by hand-checked days, invariants, batch = loop, and conformance with the
tests/fakes.py test double (a second transition implementation is allowed only with this test)."""

from dataclasses import replace

import numpy as np
import pytest
from fakes import FakeEnv, tiny_config
from hypothesis import given, settings
from hypothesis import strategies as st

import perinv.env
from perinv.env.dynamics import EVENT_ORDER_VERSION, clip_orders, initial_state, step
from perinv.utils.run_io import event_order_version

CFG1 = replace(tiny_config(n_skus=1, capacity=20), horizon=30)   # L=3, tau=2, p/c/h/w/b = 10/4/0.1/2/2


def one(cfg, inv, pipe, q, d):
    """step() for a single unbatched state."""
    return step(np.array([inv]), np.array([pipe]), np.array([q]), np.array([d]), cfg)


# ---------------------------------------------------------------- hand-checked days (N=1, L=3, P=2)

def test_ordinary_day():
    # occ 10 -> R 10, order 5 fits; arrive 3 -> inv [2,1,3]; demand 4 FIFO takes 2,1,1 -> [0,0,2];
    # expire 0, age -> [0,2,0]; held 2; pipe [4,5]; r = 40 - 20 - 0.2
    tr = one(CFG1, [[2, 1, 0]], [[3, 4]], [5], [4])
    assert tr.inv.tolist() == [[[0, 2, 0]]] and tr.pipe.tolist() == [[[4, 5]]]
    assert (tr.sold, tr.lost, tr.expired, tr.held, tr.ordered) == ([[4]], [[0]], [[0]], [[2]], [[5]])
    assert not tr.clipped[0] and tr.reward[0] == pytest.approx(19.8)


def test_oldest_stock_expires_after_fifo():
    # 3 units about to expire, demand 1 sells one of them, 2 expire: r = 10 - 2*2
    tr = one(CFG1, [[3, 0, 0]], [[0, 0]], [0], [1])
    assert (tr.sold, tr.expired, tr.held) == ([[1]], [[2]], [[0]]) and tr.reward[0] == pytest.approx(6.0)
    assert {k: float(v[0]) for k, v in tr.breakdown.items()} == pytest.approx(
        {"revenue": 10, "ordering": 0, "holding": 0, "expiry": 4, "lost_sales": 0})


def test_stockout_is_lost_sales():
    tr = one(CFG1, [[0, 0, 0]], [[0, 0]], [0], [3])
    assert (tr.sold, tr.lost) == ([[0]], [[3]]) and tr.reward[0] == pytest.approx(-6.0)


def test_todays_arrival_can_be_sold_today():
    tr = one(CFG1, [[0, 0, 0]], [[2, 0]], [0], [5])
    assert (tr.sold, tr.lost, tr.held) == ([[2]], [[3]], [[0]])


def test_order_arrives_after_lead_time_and_lives_L_days():
    """Ordered on day 0 with tau = 2: in transit on days 1-2, sold from day 2; unsold, it expires end of day 4."""
    inv, pipe = np.zeros((1, 1, 3), np.int64), np.zeros((1, 1, 2), np.int64)
    history = []
    for t in range(6):
        tr = step(inv, pipe, np.array([[5 if t == 0 else 0]]), np.zeros((1, 1), np.int64), CFG1)
        history.append((int(tr.held[0, 0]), int(tr.expired[0, 0])))
        inv, pipe = tr.inv, tr.pipe
    assert history == [(0, 0), (0, 0), (5, 0), (5, 0), (0, 5), (0, 0)]


def test_freshest_bucket_is_empty_at_decision_time():
    """SPEC §2: the DP state drops inv[:, L-1] because it is always 0 at decision time."""
    rng = np.random.default_rng(0)
    inv, pipe = np.zeros((64, 1, 3), np.int64), np.zeros((64, 1, 2), np.int64)
    for _ in range(40):
        tr = step(inv, pipe, rng.integers(0, 21, (64, 1)), rng.poisson(4, (64, 1)), CFG1)
        inv, pipe = tr.inv, tr.pipe
        assert (inv[:, :, -1] == 0).all()


# ---------------------------------------------------------------- initial state (D4)

def test_initial_state_fills_the_pipeline_by_expected_demand():
    cfg = replace(tiny_config(n_skus=2, capacity=40), demand_mean=(4.0, 5.0), season=(1.0, 0.5, 1, 1, 1, 1, 1))
    inv, pipe = initial_state(cfg)
    assert inv.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert pipe.tolist() == [[4, 2], [5, 3]]           # slot k arrives on day k: lambda * season[k]; 2.5 -> 3


def test_initial_state_is_capped_by_the_clip_rule():
    cfg = tiny_config(n_skus=2, capacity=5)            # wants [[4, 4], [4, 4]] = 16 > 5
    _, pipe = initial_state(cfg)
    assert pipe.sum() == 5 and pipe.tolist() == [[1, 1], [1, 2]]


# ---------------------------------------------------------------- clipping (D2)

@pytest.mark.parametrize("q, occ, cap, expected", [
    ([5, 5, 2], 14, 20, [2, 2, 2]),     # R 6
    ([3, 3], 15, 20, [2, 3]),           # R 5: the tie loses its unit at the lowest index
    ([1, 0, 7], 17, 20, [1, 0, 2]),
    ([3, 3, 3], 20, 20, [0, 0, 0]),     # R 0
    ([2, 1, 0], 10, 20, [2, 1, 0]),     # fits: unchanged
])
def test_clip_rule(q, occ, cap, expected):
    out, clipped = clip_orders(np.array([q]), np.array([occ]), cap)
    assert out[0].tolist() == expected and bool(clipped[0]) == (sum(q) > cap - occ)


def test_clip_leaves_input_untouched():
    q = np.array([[9, 9]])
    clip_orders(q, np.array([0]), 4)
    assert q.tolist() == [[9, 9]]


# ---------------------------------------------------------------- input checks

@pytest.mark.parametrize("kwargs, match", [
    ({"q": [[-1]]}, ">= 0"),
    ({"q": [[1.5]]}, "integer"),
    ({"d": [[1, 2]]}, "shape"),
    ({"inv": [[[20, 0, 0]]], "pipe": [[[1, 0]]]}, "exceeds capacity"),
])
def test_bad_inputs_are_rejected(kwargs, match):
    args = {"inv": [[[0, 0, 0]]], "pipe": [[[0, 0]]], "q": [[0]], "d": [[0]]} | kwargs
    with pytest.raises(ValueError, match=match):
        step(*(np.array(args[k]) for k in ("inv", "pipe", "q", "d")), CFG1)


def test_event_order_version_reaches_train_key():
    assert perinv.env.EVENT_ORDER_VERSION == EVENT_ORDER_VERSION == event_order_version() != "unversioned"


# ---------------------------------------------------------------- invariants and batching

configs = st.builds(
    lambda n, cap, L, tau, seed: (replace(tiny_config(n_skus=n, capacity=cap, horizon=30), shelf_life=L,
                                          lead_time=tau), seed),
    st.integers(1, 3), st.integers(1, 30), st.integers(1, 4), st.integers(1, 3), st.integers(0, 2**32 - 1))


def random_states(cfg, rng, b):
    """Reachable-looking states: simulate a few random days from empty."""
    inv = np.zeros((b, cfg.n_skus, cfg.shelf_life), np.int64)
    pipe = np.zeros((b, cfg.n_skus, cfg.lead_time), np.int64)
    for _ in range(int(rng.integers(0, 8))):
        tr = step(inv, pipe, rng.integers(0, cfg.capacity + 1, (b, cfg.n_skus)),
                  rng.poisson(3, (b, cfg.n_skus)), cfg)
        inv, pipe = tr.inv, tr.pipe
    return inv, pipe


@settings(max_examples=60, deadline=None)
@given(configs)
def test_invariants_and_batch_equals_loop(case):
    cfg, seed = case
    rng = np.random.default_rng(seed)
    b = 16
    inv, pipe = random_states(cfg, rng, b)
    q = rng.integers(0, cfg.capacity + 3, (b, cfg.n_skus))
    d = rng.poisson(4, (b, cfg.n_skus))
    tr = step(inv, pipe, q, d, cfg)
    occ0 = inv.sum(axis=(1, 2)) + pipe.sum(axis=(1, 2))
    assert (tr.sold + tr.lost == d).all()
    assert (tr.ordered <= q).all() and (occ0 + tr.ordered.sum(axis=1) <= cfg.capacity).all()
    assert (tr.clipped == (q.sum(axis=1) > cfg.capacity - occ0)).all()
    assert (tr.inv.sum(axis=(1, 2)) + tr.pipe.sum(axis=(1, 2)) <= cfg.capacity).all()
    stock_in = inv.sum(axis=2) + pipe[:, :, 0]                       # units that could be sold today
    assert (tr.sold + tr.expired + tr.held == stock_in).all()       # every unit is sold, expired or kept
    assert (tr.inv[:, :, -1] == 0).all()
    b_ = tr.breakdown
    assert tr.reward == pytest.approx(b_["revenue"] - b_["ordering"] - b_["holding"] - b_["expiry"]
                                      - b_["lost_sales"])
    for i in range(b):                                                # batch row i == the same day alone
        row = step(inv[i:i + 1], pipe[i:i + 1], q[i:i + 1], d[i:i + 1], cfg)
        assert np.array_equal(row.inv[0], tr.inv[i]) and np.array_equal(row.pipe[0], tr.pipe[i])
        assert row.reward[0] == tr.reward[i]


# ---------------------------------------------------------------- conformance with the test double

@settings(max_examples=60, deadline=None)
@given(configs)
def test_fake_env_conforms_to_dynamics(case):
    """tests/fakes.py:FakeEnv must produce exactly what dynamics.step produces, day by day."""
    cfg, seed = case
    rng = np.random.default_rng(seed)
    demand = rng.poisson(rng.uniform(0.5, 6), (cfg.horizon, cfg.n_skus))
    env = FakeEnv(cfg)
    _, info = env.reset(options={"demand_path": demand})
    inv, pipe = info["state"]["inv"][None], info["state"]["pipe"][None]
    for t in range(cfg.horizon):
        q = rng.integers(0, cfg.capacity + 3, cfg.n_skus)
        tr = step(inv, pipe, q[None], demand[t][None], cfg)
        _, reward, _, _, info = env.step(q)
        for key in ("sold", "lost", "expired", "held", "ordered"):
            assert np.array_equal(info[key], getattr(tr, key)[0]), (t, key)
        assert info["clipped"] == bool(tr.clipped[0])
        assert reward == pytest.approx(tr.reward[0], abs=1e-9)
        assert np.array_equal(info["state"]["inv"], tr.inv[0]) and np.array_equal(info["state"]["pipe"], tr.pipe[0])
        inv, pipe = tr.inv, tr.pipe
