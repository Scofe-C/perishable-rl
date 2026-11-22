# Project status

> Keep this current. Whoever finishes a step updates "Current step" and adds one line to "History".
> Agents: read this first, then [SPEC.md](SPEC.md). Project summary: [README.md](../README.md).

**Last updated:** 2026-11-22 · **Phase:** Week 1. The M3 pipeline is done. The M1 env and DP start now on
branch `zhengxin-m1-env` (Zhengxin covers M1 as well as M3).

## Current step
**M1 core is built (U1–U3) and measured (U4).** Open decisions from U4 (DECISIONS.md): capacities from O_ref,
D4 warm start, VecEnv choice. Then the thinnest end-to-end run once M2's `train.build` exists.

Blockers:
1. **SPEC.md is a draft.** D1–D7 (DECISIONS.md) are written in and need the other two members' confirmation.
2. **No training or (s,S)/newsvendor builders yet**: RL and heuristic jobs show `blocked (no builder yet)`.
   DP jobs (groups A, DP1/DP3/DP13) can run now.
3. **Repo admin:** the CODEOWNERS handles are placeholders, and CI has not run on Python 3.15.

Next, by owner (README has the owner table):
- **M1:** see the checklist below.
- **M2:** `agents/factory.py`, `agents/sb3_policy.py` (it must return order quantities; DECISIONS 2026-11-22),
  `train/train.py` + `callbacks.py`, and the (s,S) and newsvendor baselines, all as `build(ctx) -> Policy`
  (WAREHOUSE §7).
- **M3:** training on demand shocks
  (unit D) once M2's `train.build` exists.
- **All, by Friday:** `scripts/smoke_e2e.py` (N=1 → PPO 10k steps → 5 paths → plot) and `bench_vecenv.py`,
  then set `train_env_steps` in `configs/budget.yaml`.

## M1 checklist
Must have:
1. [x] U1–U3: dynamics, env + wrappers, exact DP (all M1 pending guards replaced by real tests).
2. [ ] **O_ref → C per κ**: measured with the provisional stand-in (`scripts/measure_oref.py`; DECISIONS.md);
   applying the new capacities is open. Remeasure with M2's (s,S).
3. [x] **Exact DP** (no truncation: the d ≥ I row weighted by P(D ≥ I), lost sales −b·E[(D−I)⁺]); equals brute force to 1e-9 and the simulated policy within 4 SE.
4. [x] **DP with seasonality and fractional λ** (calibrated N=1 configs solve; simulation agrees).

Should have:
5. [x] **Cold-start effect** measured (`dp.backward_induction.daily_profile`): empty start costs ≈ 15–18 % of V0; warm-start proposal in DECISIONS.md.
6. [x] **Benchmarks** (`scripts/bench_vecenv.py`, Windows): PPO ≈ 2.9k env steps/s with DummyVecEnv×4 (1M steps ≈ 6 min); rerun on Linux.

Optional:
7. [ ] BSP-low-EW baseline (Haijema & Minner 2019), a strong age-aware reference for N ≥ 2.
8. [ ] Cross-check the DP against mdpax (Farrington et al.) on a small instance.

Done by decision: the mapping endpoints (D1) and the clipping rule (D2).

## What exists
| Status | What |
|---|---|
| Working + tested | `env/` (config, dynamics, env, wrappers), `dp/` + `baselines/dp_policy.py` (exact DP, cached), `data/` (CRN, calibration, scenarios), `eval/` (evaluate, writer, warehouse, metrics, stats), `runner.py`, `utils/`, `baselines/base.py`; scripts `make_crn_paths`, `solve_dp`, `measure_oref` (provisional), `bench_vecenv`, `make_calibrated_configs`, `make_shock_configs`, `run_matrix`, `analyze`; infra data platform (`infra/`) |
| Stub (docstring only) | `baselines/{s_S,newsvendor}.py`, `agents/`, `train/`; script `smoke_e2e` |
| Provisional values | capacities (`κ·20·N`), cost parameters (SPEC §1) |
| Tests | 177 passed, 2 skipped (M2's pending pitfall guards), plus 2 slow tests (`pytest -m slow`) |

Platform notes:
- Final experiments run on Linux or WSL2. Development also works on native Windows (docs/infra/RUN.md).
- On Windows without MSVC, install with `PYYAML_FORCE_LIBYAML=0` (there is no cp315 wheel).
- SubprocVecEnv and the runner use spawn: always run real script files with an `if __name__ == "__main__":` guard.

## History
- 2026-11-22 (M1): U4 measurements. O_ref with a provisional order-up-to stand-in (O_ref = 15 per SKU at λ = 4); `daily_profile` (exact daily expectations from the DP) shows the empty start costs ≈ 15–18 % of V0 and that a pre-filled pipeline removes it; `bench_vecenv.py`: training is model-bound (PPO ≈ 2.9k steps/s, DummyVecEnv×4). Capacities, warm start and VecEnv are proposed, not applied. CRN paths for all configs generated locally (data/crn, 315 KB, not committed yet).
- 2026-11-22 (M1): U3 exact N=1 DP. `dp/state_space.py` (C(C+L−1+P, L−1+P) states, searchsorted index), `dp/backward_induction.py` (one batched `dynamics.step` call builds the table; tail-lumped exact expectation; `evaluate_policy` for exact values of any policy), `baselines/dp_policy.py` (builder, cache `results/dp/<config_hash>.npz`), `scripts/solve_dp.py`. Tests: brute force to 1e-9 (mutation-checked), table rows = env steps, DP ≥ every constant policy (exact), V0 = simulated mean within 4 SE (also seasonal/calibrated), a runner DP job with pct_of_dp = 1. Found and closed a race: parallel DP jobs on one config (shocks.yaml A_*) shared `utils.atomic`'s single `<name>.tmp` (old code failed 4/5 runs); the cache now uses per-writer temp files and treats an unreadable file as a miss. C=20 solves in 0.35 s. V0 (empty start): n1_k06 507.4, n1_k10 579.9, frn_m1_n1_k06 233.6, frn_m3_n1_k06 402.4.
- 2026-11-22 (M1): U2 `PerishableInventoryEnv` (D3–D6) and wrappers `CapacityShareAction` (D1), `DiscreteOrder`; the mapping is the pure function `capacity_share_orders`, which M2's adapter should reuse. Tests: gymnasium and SB3 `check_env`, exact CRN replay, seeded training demand, capacity, masks reaching MaskablePPO through Monitor/DummyVecEnv/SubprocVecEnv, D1 endpoints for every R ≤ 1000, identical `evaluate` results to the test double, and a runner run on the real env. D7: softmax(4·a[1:]), because the plain split capped one SKU at 65 % for N=5.
- 2026-11-22 (M1): U1 `env/dynamics.py`, the single batched transition (SPEC §3, D2 clipping, `EVENT_ORDER_VERSION = "1"`, so every train_key changes once). Tests: hand-checked days, invariants, batch = loop, and a hypothesis conformance test of `tests/fakes.py` (mutation-checked). Speed: 122 ns/row in batch (DP table), 38 µs per single step.
- 2026-11-22 (M1): Branch `zhengxin-m1-env`. SPEC decisions D1–D6 and the DP state-count correction; docs trimmed.
- 2026-11-22 (M3): YAML `x-` annotations; M1 checklist from the team draft.
- 2026-11-22 (M3): Robustness (`tests/test_faults.py`). Each parallel job runs in its own process, so a hard crash fails only that job (`WorkerCrashed`). NaN rewards are rejected. Policy actions are checked. A missing CRN path blocks the run with the full fix command. The config generators are all-or-nothing and drop stale files.
- 2026-11-22 (M3): Demand-shock scenarios, eval-only (`data/scenarios.py`, `configs/experiments/shocks.yaml`, `metrics.paired_worlds`).
- 2026-11-14 (M3): Cross-evaluation (`eval_envs`: build on A, score on B, reuse A's model); fixed a Windows manifest-rename race.
- 2026-11-14 (M3): Real-data calibration: infra C1 → `configs/env/calibrated/` (12 configs) and group D.
- 2026-11-14 (M3): Warehouse: DuckDB views (`eval/sql/views.sql`), IQM + bootstrap CIs, `scripts/analyze.py`.
- 2026-11-14 (M3): Runner: idempotent matrix (`run_key`), skip done, K parallel workers, resume after a kill.
- 2026-11-14 (M3): `run_io` (keys, manifest, status lifecycle), `evaluate` + validated Parquet writer, `utils/atomic`.
- 2026-11-14 (M3): Added duckdb 2.0.0; verified the pinned env on native Windows.
- 2026-11-13: Repo skeleton from the team draft; Python 3.15.2 with pinned requirements; rliable dropped.
