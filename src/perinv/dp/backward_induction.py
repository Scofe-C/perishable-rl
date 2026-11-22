"""Exact finite-horizon backward induction for N = 1 (SPEC §3, undiscounted, no salvage).

    sol = solve(cfg)        # V[t, s] (expected profit from day t on), pi[t, s] (optimal order)

Exact, with no demand truncation. Let I be the stock on hand after today's arrival (I ≤ C). Every demand
d ≥ I sells I and leads to the same next state; it differs only by the lost-sales penalty b·(d − I). So the
transition table has rows d = 0 … I, the d = I row carries the weight P(D ≥ I), and the penalty enters
analytically as −b·E[(D − I)⁺] = −b·(λ·P(D ≥ I) − I·P(D ≥ I+1)).

Every row is produced by perinv.env.dynamics.step (one batched call), so the DP and the env cannot disagree
on the event order. The table does not depend on t; seasonality only changes the per-day demand weights.
Ties between orders go to the smallest order (np.argmax), so the policy is deterministic.

Owner: Member 1. Contract: docs/SPEC.md §2, §3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from perinv.dp import state_space
from perinv.dp.state_space import StateSpace
from perinv.env import dynamics
from perinv.env.config import EnvConfig


@dataclass(frozen=True)
class Table:
    """All (state, feasible order, demand) rows for one config. pair = (state, order); row = (pair, d)."""

    space: StateSpace
    pair_state: np.ndarray    # int64[Q]
    pair_order: np.ndarray    # int64[Q]
    row_pair: np.ndarray      # int64[R]
    row_demand: np.ndarray    # int64[R], 0 … I; the d = I row stands for "d ≥ I"
    row_on_hand: np.ndarray   # int64[R], I of the row's state
    row_reward: np.ndarray    # float64[R], reward with lost = 0 (true for every d ≤ I)
    row_next: np.ndarray      # int64[R], next state index


@dataclass(frozen=True)
class Solution:
    space: StateSpace
    V: np.ndarray             # float64[T+1, S]; V[T] = 0
    pi: np.ndarray            # int64[T, S]

    def start_index(self, cfg: EnvConfig) -> int:
        """State index of the SPEC §3 initial state (D4 warm start)."""
        inv, pipe = dynamics.initial_state(cfg)
        return int(self.space.index(self.space.from_env(inv[0], pipe[0])))

    def initial_value(self, cfg: EnvConfig) -> float:
        """V[0] of the SPEC §3 initial state: the expected profit of an episode under the optimal policy."""
        return float(self.V[0, self.start_index(cfg)])


def _ranges(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(owner, offset): for each owner i, offsets 0 … lengths[i]-1, flattened."""
    owner = np.repeat(np.arange(len(lengths)), lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return owner, offset


def build_table(cfg: EnvConfig) -> Table:
    if cfg.n_skus != 1:
        raise ValueError(f"the exact DP is for N = 1 only, got N = {cfg.n_skus}")
    space = state_space.build(cfg.capacity, cfg.shelf_life, cfg.lead_time)
    pair_state, pair_order = _ranges(cfg.capacity - space.occupancy() + 1)     # q = 0 … C − occ
    on_hand = space.on_hand_after_arrival()
    row_pair, row_demand = _ranges(on_hand[pair_state] + 1)                    # d = 0 … I
    s = pair_state[row_pair]
    inv, pipe = space.to_env(space.states[s])
    tr = dynamics.step(inv, pipe, pair_order[row_pair][:, None], row_demand[:, None], cfg)
    assert not tr.clipped.any() and (tr.lost == 0).all()
    nxt = space.index(space.from_env(tr.inv[:, 0, :], tr.pipe[:, 0, :]))
    return Table(space, pair_state, pair_order, row_pair, row_demand, on_hand[s], tr.reward, nxt)


def demand_terms(lam: float, capacity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For I = 0 … C: pmf P(D = I), tail P(D ≥ I), and E[(D − I)⁺], for D ~ Poisson(lam)."""
    i = np.arange(capacity + 1)
    pmf = poisson.pmf(i, lam)
    tail = poisson.sf(i - 1, lam)                 # P(D ≥ I) = P(D > I − 1)
    excess = lam * tail - i * poisson.sf(i, lam)  # E[(D − I)⁺] = λ·P(D ≥ I) − I·P(D ≥ I+1)
    return pmf, tail, np.maximum(excess, 0.0)


def _backup(cfg: EnvConfig, table: Table):
    """q_values(t, V_next) -> float64[S, C+1]: expected value of each (state, order) on day t, −inf if infeasible."""
    space, S, C = table.space, len(table.space), cfg.capacity
    lam_t = cfg.demand_mean[0] * np.asarray(cfg.season)[np.arange(cfg.horizon) % 7]
    b = cfg.lost_sales_cost[0]
    is_tail = table.row_demand == table.row_on_hand
    pair_on_hand = space.on_hand_after_arrival()[table.pair_state]
    terms = {float(lam): demand_terms(float(lam), C) for lam in np.unique(lam_t)}   # ≤ 7 distinct λ

    def q_values(t: int, V_next: np.ndarray) -> np.ndarray:
        pmf, tail, excess = terms[float(lam_t[t])]
        w = np.where(is_tail, tail[table.row_on_hand], pmf[table.row_demand])
        value = np.bincount(table.row_pair, weights=w * (table.row_reward + V_next[table.row_next]),
                            minlength=len(table.pair_state)) - b * excess[pair_on_hand]
        grid = np.full((S, C + 1), -np.inf)
        grid[table.pair_state, table.pair_order] = value
        return grid

    return q_values


def solve(cfg: EnvConfig, table: Table | None = None) -> Solution:
    table = table or build_table(cfg)
    S, T = len(table.space), cfg.horizon
    q_values = _backup(cfg, table)
    V = np.zeros((T + 1, S))
    pi = np.zeros((T, S), np.int64)
    for t in range(T - 1, -1, -1):
        grid = q_values(t, V[t + 1])
        pi[t] = np.argmax(grid, axis=1)
        V[t] = grid[np.arange(S), pi[t]]
    return Solution(table.space, V, pi)


def evaluate_policy(cfg: EnvConfig, pi: np.ndarray, table: Table | None = None) -> np.ndarray:
    """Exact expected profit V[t, s] of a fixed policy pi[t, s] (orders above R are clipped first, as the env does)."""
    table = table or build_table(cfg)
    S, T = len(table.space), cfg.horizon
    q_values = _backup(cfg, table)
    feasible = cfg.capacity - table.space.occupancy()
    V = np.zeros((T + 1, S))
    for t in range(T - 1, -1, -1):
        V[t] = q_values(t, V[t + 1])[np.arange(S), np.minimum(pi[t], feasible)]
    return V


@dataclass(frozen=True)
class DailyProfile:
    reward: np.ndarray      # float64[T]: expected reward on day t
    occupancy: np.ndarray   # float64[T]: expected decision-time occupancy on day t


def daily_profile(cfg: EnvConfig, pi: np.ndarray, start: np.ndarray, table: Table | None = None) -> DailyProfile:
    """Exact day-by-day expectations of policy pi from a start distribution over states (float[S], sums to 1).
    sum(reward) equals the policy's value from that start; used to measure the cold-start effect (D4)."""
    table = table or build_table(cfg)
    space, S, C, T = table.space, len(table.space), cfg.capacity, cfg.horizon
    lam_t = cfg.demand_mean[0] * np.asarray(cfg.season)[np.arange(T) % 7]
    b = cfg.lost_sales_cost[0]
    pair_of = np.full((S, C + 1), -1, np.int64)
    pair_of[table.pair_state, table.pair_order] = np.arange(len(table.pair_state))
    is_tail = table.row_demand == table.row_on_hand
    feasible = C - space.occupancy()
    on_hand = space.on_hand_after_arrival()
    mu = np.asarray(start, np.float64)
    reward, occupancy = np.zeros(T), np.zeros(T)
    for t in range(T):
        pmf, tail, excess = demand_terms(float(lam_t[t]), C)
        chosen = np.zeros(len(table.pair_state))
        chosen[pair_of[np.arange(S), np.minimum(pi[t], feasible)]] = mu      # probability of each (state, order)
        w = np.where(is_tail, tail[table.row_on_hand], pmf[table.row_demand]) * chosen[table.row_pair]
        reward[t] = w @ table.row_reward - b * (mu @ excess[on_hand])
        occupancy[t] = mu @ space.occupancy()
        mu = np.bincount(table.row_next, weights=w, minlength=S)
    return DailyProfile(reward, occupancy)
