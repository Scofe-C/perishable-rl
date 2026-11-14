# SPEC — shared contracts

> **Status: DRAFT, not yet ratified.** All three members review it in the week-1 meeting. After that
> it is frozen. From then on, any change needs approval from all three (CODEOWNERS) and an entry in
> [DECISIONS.md](DECISIONS.md).
>
> Every algorithm, baseline and the DP depend on this file. If the env and the DP read it
> differently, both test suites can pass while the comparison is silently wrong.

## 1. Parameters

| Symbol | Meaning | Default (PROPOSED) |
|---|---|---|
| N | number of SKUs | 1, 2, 5 (core); 10, 20 (extension) |
| C | shared capacity, units, summed over SKUs | derived from κ (§1.1) |
| L | shelf life in days (age buckets) | 3 |
| τ | lead time in days, τ ≥ 1; P = τ pipeline slots | 2 |
| λ_i | mean daily Poisson demand of SKU i | 4 |
| season[0..6] | day-of-week multipliers (mean 1) | all 1 (off) |
| T | horizon in days | 30 |
| p_i | sale price per unit | 10 |
| c_i | ordering cost per unit | 4 |
| h_i | holding cost per unit per day | 0.1 |
| w_i | expiry penalty per expired unit (on top of c_i already paid) | 2 |
| b_i | lost-sales penalty per unit of unmet demand | 2 |
| γ | discount | 1 (finite horizon, undiscounted) |

### 1.1 Capacity tightness κ
`C = round(κ · O_ref)`. `O_ref` is the mean total occupancy (on hand + in transit + new orders) of
the unconstrained (s,S) policy on the `tune` paths. **Week 1 TODO:** measure O_ref per N and record
it in DECISIONS.md. κ ∈ {0.6, 1.0}.

## 2. State
At the start of day t:
- `inv[N, L]`: on-hand stock by remaining shelf life. Index 0 expires at the end of today, and index L−1 is the freshest.
- `pipe[N, P]` with P = τ: orders placed on days t−τ … t−1. `pipe[:, 0]` arrives today.
- `t` (0 … T−1), from which day-of-week = `t mod 7`.

Occupancy is `occ = sum(inv) + sum(pipe)`. Today's arrival still counts as in transit.

## 3. Event order within day t (the single source of truth)
1. **Decide.** Observe the state and choose the integer order `q[N] ≥ 0`.
   The order is feasible iff `occ + sum(q) ≤ C`.
   If it is infeasible, the env **clips** it to the remaining capacity `R = C − occ`, removing units
   from the largest orders first, and sets `info["clipped"] = True`.
2. **Arrive.** `inv[:, L-1] += pipe[:, 0]`.
3. **Demand.** Draw `d_i ~ Poisson(λ_i · season[t mod 7])`, or read it from the CRN path. Issue it
   FIFO from bucket 0 upward. `sold = min(d, sum(inv))` and `lost = d − sold`.
4. **Expire.** The remaining `inv[:, 0]` becomes `expired`. Shift ages so that `inv[:, k] ← inv[:, k+1]` and `inv[:, L-1] ← 0`.
5. **Hold.** `held = sum(inv)` after expiry.
6. **Pipeline.** Shift `pipe[:, k] ← pipe[:, k+1]` and set `pipe[:, P-1] ← q`.
7. **Reward.** `r = Σ_i [p·sold − c·q − h·held − w·expired − b·lost]`.
8. `t ← t+1`. The episode is **truncated** at `t = T`, never terminated. There is no salvage value for leftover stock.

Initial state (PROPOSED): `inv = 0, pipe = 0`. Revisit if the cold-start days distort results.

## 4. Action mappings

### 4.1 Box (PPO, A2C, SAC): one shared wrapper `CapacityShareAction`
- `action ∈ Box(-1, 1, shape=(N+1,))`
- `R = C − occ`
- `share = sigmoid(4 · a[0])`. The ×4 is required: a raw `sigmoid(a)` stays in [0.27, 0.73].
- `total = clip(floor(share · R + 0.5), 0, R)`
- `w = softmax(a[1:])`, and `q` = largest-remainder rounding of `w · total`, so `sum(q) == total` exactly.
- Required: `a[0] = −1 → total = 0` and `a[0] = +1 → total = R` after rounding. Make sure the ×4 actually achieves this; if it doesn't, raise the scale and record the change in DECISIONS.md.

### 4.2 Discrete (DQN, MaskablePPO), N = 1 only
- `action ∈ Discrete(C + 1)`, and `q = action`.
- The env exposes `action_masks() -> bool[C+1]` where mask[k] is `k ≤ R`. It lives on the env, not
  on a wrapper, so it works under SubprocVecEnv.
- DQN has no masks, so it relies on clipping (§3.1). Clipped-step frequency is logged.

## 5. Observation (PROPOSED)
`obs = concat(inv.flatten() / C, pipe.flatten() / C, onehot(t mod 7), [t / T], [occ / C])` as float32.
Algorithms may add VecNormalize on top. The raw state is always available in `info["state"]`.

## 6. `info` dict from `step()`
`sold, demand, lost, expired, ordered, held, occupancy, clipped, reward_breakdown{revenue, ordering,
holding, expiry, lost_sales}, state{inv, pipe, t}`

## 7. CRN demand paths
- Location: `data/crn/<demand_hash>/{tune,test}.npz` + `manifest.json`.
- `demand_hash` is the sha256 of the canonical JSON of `{N, lambda, season, T}`, first 12 hex characters.
- npz arrays: `demand: int32[n_paths, T, N]` and `path_seed: int64[n_paths]`.
- Path k of split s is drawn with `np.random.default_rng([root_seed, SPLIT_ID[s], k])`, where `SPLIT_ID = {tune: 0, test: 1}`. The two splits never overlap.
- Sizes: `tune` = 50 paths (hyperparameter tuning and (s,S) grid search) and `test` = 100 paths (final evaluation only).
- `manifest.json` records the params, `root_seed`, `version`, and the sha256 of each npz file.
- Evaluation: `env.reset(options={"demand_path": demand[k]})`. Stochastic policies and random baselines are seeded with `k`.

## 8. Policy interface
```python
class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...  # int64[N], feasible
```
Every agent goes through the same `evaluate()`: DP, (s,S), newsvendor, and SB3 models (with frozen
VecNormalize: `training=False, norm_reward=False`, `deterministic=True`, and masks applied for
MaskablePPO).
