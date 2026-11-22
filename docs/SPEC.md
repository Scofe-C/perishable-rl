# SPEC — shared contracts

> **Status: DRAFT.** Decisions D1–D7 (2026-11-22, [DECISIONS.md](DECISIONS.md)) are written in below and wait
> for team confirmation. Once ratified, any change needs all three members and a DECISIONS.md entry.
>
> The env, the DP and every agent depend on this file. If the env and the DP read it differently, both
> test suites can pass while the comparison is silently wrong.

## 1. Parameters

| Symbol | Meaning | Default (proposed) |
|---|---|---|
| N | number of SKUs | 1, 2, 5 (core); 10, 20 (extension) |
| C | shared capacity in units, summed over SKUs | from κ (§1.1) |
| L | shelf life in days (age buckets) | 3 |
| τ | lead time in days, τ ≥ 1; P = τ pipeline slots | 2 |
| λ_i | mean daily Poisson demand of SKU i | 4 |
| season[0..6] | day-of-week multipliers (mean 1) | all 1 (off) |
| T | horizon in days | 30 |
| p, c, h, w, b | price, order cost, holding cost/day, expiry penalty (on top of c), lost-sales penalty, per unit | 10, 4, 0.1, 2, 2 |
| γ | discount | 1 (finite horizon, undiscounted) |

### 1.1 Capacity tightness κ
`C = round(κ · O_ref)`, with κ ∈ {0.6, 1.0}. O_ref is the mean total occupancy (on hand + in transit + new
orders) of the unconstrained (s,S) policy on the `tune` paths. Until M2's (s,S) exists, O_ref is measured
with a provisional order-up-to stand-in and marked as such. The configs use `κ·20·N` until then.

## 2. State
At the start of day t:
- `inv[N, L]`: on-hand stock by remaining shelf life. Index 0 expires at the end of today; L−1 is the freshest.
- `pipe[N, P]`: orders placed on days t−τ … t−1. `pipe[:, 0]` arrives today.
- `t` (0 … T−1); day of week = `t mod 7`.

Occupancy `occ = sum(inv) + sum(pipe)`. Today's arrival still counts as in transit.

At decision time `inv[:, L−1]` is always 0: arrivals enter it in step 2, and step 4 empties it. The DP state is
therefore `(inv[:, :L−1], pipe)` with `occ ≤ C`, which gives **C(C+L−1+P, L−1+P)** states for N=1 (C=20, L=3,
P=2: 10,626).

## 3. Event order within day t (the single source of truth)
1. **Decide.** Observe the state and choose integer orders `q[N] ≥ 0`. Feasible iff `occ + sum(q) ≤ C`.
   If infeasible, the env clips (D2) and sets `info["clipped"] = True`:
   ```
   R = C − occ
   while sum(q) > R:
       i = argmax(q)          # ties: the lowest SKU index
       q[i] −= 1
   ```
2. **Arrive.** `inv[:, L−1] += pipe[:, 0]`.
3. **Demand.** `d_i ~ Poisson(λ_i · season[t mod 7])`, or read from the CRN path. Issue FIFO from bucket 0
   upward. `sold = min(d, sum(inv))`, `lost = d − sold`.
4. **Expire.** The remaining `inv[:, 0]` is `expired`. Shift ages: `inv[:, k] ← inv[:, k+1]`, `inv[:, L−1] ← 0`.
5. **Hold.** `held = sum(inv)` after expiry.
6. **Pipeline.** Shift `pipe[:, k] ← pipe[:, k+1]`, set `pipe[:, P−1] ← q`.
7. **Reward.** `r = Σ_i [p·sold − c·q − h·held − w·expired − b·lost]`.
8. `t ← t+1`. At `t = T` the episode is **truncated** (`truncated=True`); it is never terminated. No salvage value.

- **Initial state (D4):** `inv = 0, pipe = 0` (provisional, until the cold-start effect is measured with the DP).
- **Demand without a CRN path (D5):** `reset()` without `options["demand_path"]` draws each day's demand from
  `self.np_random` (seeded by `reset(seed=...)`). This is the training mode.

## 4. Action mappings

### 4.0 Base env (D6)
`action_space = MultiDiscrete([C+1] * N)`: the integer orders `q[N]`, clipped as in §3.

### 4.1 Box (PPO, A2C, SAC): one shared wrapper `CapacityShareAction` (D1)
- `action ∈ Box(-1, 1, shape=(N+1,))`, `R = C − occ`
- `share = (σ(4·a[0]) − σ(−4)) / (σ(4) − σ(−4))`. This rescaled sigmoid maps a[0] = −1 to exactly 0 and +1 to
  exactly 1, for every R. The plain `σ(4·a[0])` fails both endpoints from R = 28 (measured), and a raw
  `σ(a)` would stay in [0.27, 0.73].
- `total = clip(floor(share · R + 0.5), 0, R)`
- `w = softmax(4 · a[1:])` (D7: plain softmax on [−1, 1] caps one SKU at 65 % for N = 5); `q` = largest-remainder
  rounding of `w · total` (leftover units to the largest fractions, ties to the lowest index), so `sum(q) == total`.
- Implemented once as `perinv.env.wrappers.capacity_share_orders(a, R)`; the SB3 adapter reuses it.

### 4.2 Discrete (DQN, MaskablePPO), N = 1 only: wrapper `DiscreteOrder`
- `action ∈ Discrete(C + 1)`, `q = [action]`.
- The env (not a wrapper) exposes `action_masks() -> bool[C+1]`, mask[k] = `k ≤ R`, so it works under
  SubprocVecEnv. DQN has no masks and relies on clipping; the clipped-day count is logged.

## 5. Observation (proposed)
`obs = concat(inv.flatten() / C, pipe.flatten() / C, onehot(t mod 7), [t / T], [occ / C])`, float32.
Algorithms may add VecNormalize. The raw state is always in `info["state"]`.

## 6. `info` from `reset()` and `step()`
- `reset()`: `state{inv, pipe, t}`.
- `step()`: per-SKU arrays `sold, demand, lost, expired, ordered, held`; `occupancy` = decision-time
  occupancy, before the order (D3); `clipped`; `reward_breakdown{revenue, ordering, holding, expiry,
  lost_sales}`; `state{inv, pipe, t}` after the step.
- `perinv.env.dynamics.EVENT_ORDER_VERSION` changes whenever §3 changes (it is part of `train_key`).

## 7. CRN demand paths
- Location: `data/crn/<demand_hash>/{tune,test}.npz` + `manifest.json`.
- `demand_hash` = first 12 hex characters of the sha256 of the canonical JSON of `{N, lambda, season, T}`
  (plus `scenario` when set; see `perinv.data.scenarios`).
- npz arrays: `demand: int32[n_paths, T, N]`, `path_seed: int64[n_paths]`.
- Path k of split s is drawn with `np.random.default_rng([root_seed, SPLIT_ID[s], k])`, `SPLIT_ID = {tune: 0,
  test: 1}`, so the splits never overlap.
- Sizes: `tune` = 50 paths (tuning and (s,S) grid search), `test` = 100 paths (final evaluation only).
- Evaluation: `env.reset(options={"demand_path": demand[k]})`; stochastic policies are seeded with k.

## 8. Policy interface
```python
class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...  # int64[N] order quantities
```
Every agent goes through the same `evaluate()`: DP, (s,S), newsvendor and SB3 models (frozen VecNormalize:
`training=False, norm_reward=False`; `deterministic=True`; masks applied for MaskablePPO). `evaluate()` rejects
actions that are not N whole, finite, non-negative numbers. Orders above R are legal; the env clips them.
