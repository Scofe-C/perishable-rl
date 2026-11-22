# Decision log

Newest first. One entry per change to SPEC.md, the budgets, the fairness rules or a shared dependency.
"Team to confirm" means proposed by one member and still waiting for the other two.

| Date | Who / status | Decision | Why |
|---|---|---|---|
| 2026-11-22 | Zhengxin — **team to confirm**; applied | **Capacities from O_ref** (provisional order-up-to stand-in, measured with the warm start; `configs/env/oref_provisional.json`): O_ref = 16/32/79 for N = 1/2/5, so C = 10/16, 19/32, 47/79 for κ = 0.6/1.0, replacing `κ·20·N` (12/20, 24/40, 60/100). Calibrated configs share one capacity for m1 and m3 (D13 needs it), from the m3 O_ref 13/27/36: 8/13, 16/27, 22/36 | SPEC §1.1. Under order-up-to the post-order position is exactly S, so O_ref = Σ S_i. N=5 got [16,16,16,16,15] for identical SKUs: S = 15 and 16 are within the noise of 50 tune paths. Remeasure with M2's (s,S) |
| 2026-11-22 | Zhengxin — **team to confirm**; applied | **D4 (replaces "empty start"):** warm start, on hand empty and `pipe[i, k] = floor(λ_i · season[k] + 0.5)`, capped at C by the clip rule (SPEC §3). `EVENT_ORDER_VERSION` → "2" | Exact DP (n1_k06, C = 12): an empty start forces stockouts on days 0–1 (−32, −24) and costs about 85 (≈ 15–18 % of V0) over the first 5 days against the mid-horizon rate; the warm start removes it (shortfall 1.4). The end-of-horizon effect stays; it is symmetric across policies |
| 2026-11-22 | Zhengxin — **team to confirm** (measured on Windows) | VecEnv for training: **DummyVecEnv, n_envs = 4** (`scripts/bench_vecenv.py`) | PPO learn() is about 2.9k env steps/s with Dummy×4, versus 2.6k with Subproc×4 and 1.9k with Dummy×1; the env alone runs 15k steps/s, so the model dominates. Subproc only adds processes next to the runner's parallel jobs. Rerun on the Linux machine before fixing `train_env_steps` |
| 2026-11-22 | Zhengxin — **team to confirm** | **D7** Box split: `w = softmax(4 · a[1:])` instead of `softmax(a[1:])` (SPEC §4.1; `perinv.env.wrappers.SPLIT_SCALE = 4`) | Agent actions live in [−1, 1], so plain softmax caps one SKU's weight at e/(e + (N−1)/e): 88 % for N = 2, 65 % for N = 5. With R = 60, N = 5, no action can order more than 39 units of one SKU. Scale 4 (same as D1) lifts the cap to 99.9 % and keeps a[1:] = 0 an even split |
| 2026-11-22 | Zhengxin (M1+M3) — **team to confirm** | **D1** Box mapping uses the rescaled sigmoid `share = (σ(4a₀) − σ(−4)) / (σ(4) − σ(−4))` (SPEC §4.1). This is a new formula, not the "raise the scale" fallback SPEC anticipated | Measured: plain `σ(4a₀)` misses both endpoints from R = 28 (R = 60 gives 1 and 59). The rescaled form is exact at both ends for every R ≤ 1000 and keeps share(0) = 0.5 |
| 2026-11-22 | Zhengxin — **team to confirm** | **D2** Clipping: while `sum(q) > R`, take one unit off the largest order, ties to the lowest SKU index (SPEC §3) | "Largest orders first" left ties open; the env and the DP must clip identically. Matches `tests/fakes.py` |
| 2026-11-22 | Zhengxin — **team to confirm** | **D3** `info["occupancy"]` is decision-time occupancy, before the order | Asked for in the old WAREHOUSE §8 (now SPEC §6); `evaluate` already reads it that way |
| 2026-11-22 | Zhengxin — **team to confirm** | **D5** `reset()` without a demand path draws Poisson demand from `self.np_random` | Training mode; evaluation always passes a CRN path |
| 2026-11-22 | Zhengxin — **team to confirm** | **D6** Base env `action_space = MultiDiscrete([C+1]*N)`; never terminated, truncated at T | The wrappers (§4.1, §4.2) translate into this one integer interface |
| 2026-11-22 | Zhengxin | N=1 DP state count is C(C+L−1+P, L−1+P), not the draft's C(C+L+P, L+P) | `inv[:, L−1]` is always 0 at decision time (SPEC §2; confirmed by simulation over four (C, L, τ)). C=20: 10,626 states, not 53,130 |
| 2026-11-22 | Zhengxin — enforces SPEC §8, **M2 please note** | `evaluate` rejects actions that are not N whole, finite, non-negative numbers; orders above R stay legal | Nothing checked the contract, and an env may silently clamp a bad order. The SB3 adapter must return order quantities, never the raw Box vector |
| 2026-11-22 | Zhengxin — extension, off by default | Optional `EnvConfig.scenario`: demand shocks on CRN **eval** paths only (`perinv.data.scenarios`, `configs/experiments/shocks.yaml`). The YAML loader rejects unknown keys except provenance keys and `x-` annotations | Robustness question without touching env dynamics or the core matrix. Configs without a scenario keep their hashes (pinned in tests) |
| 2026-11-14 | Zhengxin — **team to confirm** | Add `duckdb==2.0.0` for the evaluation warehouse | SQL straight on Parquet, no server. 2.0.0 is the only release with Python 3.15 wheels. Results are not hive-partitioned (WAREHOUSE §2) |
| 2026-11-13 | Weibo — **team to confirm** | Drop `rliable`; IQM via `scipy.stats.trim_mean(x, 0.25)`, CIs via `scipy.stats.bootstrap` | rliable needs arch < 8, which crashes with pandas 3, and there is no pandas 2 for Python 3.15 |
| 2026-11-13 | Weibo — **team to confirm** | Python 3.15 for everyone (pins resolved on 3.15.2) | Every pin has macOS arm64 and Linux x86_64 cp315 wheels or a pure-Python sdist, including `torch==2.15.0+cpu` |
| 2026-11-13 | scaffold | pip + pinned `requirements.txt`, no Docker for the team runtime; never commit pre-release pins | Nothing is deployed; pins plus `scripts/setup_linux_vm.sh` reproduce Linux runs |

## Open items for the week-1 meeting
- [ ] Ratify SPEC.md, including D1–D7
- [ ] Default cost parameters, λ, L, τ, T (SPEC §1)
- [ ] Confirm the O_ref capacities and the D4 warm start; remeasure O_ref with M2's (s,S)
- [ ] Confirm DummyVecEnv × 4 after rerunning `scripts/bench_vecenv.py` on Linux
- [ ] Training env-step budget (`configs/budget.yaml`), after timing one run
