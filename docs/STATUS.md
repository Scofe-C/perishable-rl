# Project status

> Keep this current. Whoever finishes a step updates "Current step" and adds one line to "History".
> Agents: read this first, then [SPEC.md](SPEC.md). Project summary: [README.md](../README.md).

**Last updated:** 2026-11-23 · **Phase:** Week 2. **Deadline: everything due Dec 6.** All code for the core
experiments exists; tuning is done (configs/tuned/, 22 files).

## Current step
**Final runs** (`core.yaml --official`), then the analysis. Tuning finished on 2026-11-23 (History). M2 code, the review fixes and the
smoke run landed on 2026-11-23 (History).

Run order (each command skips what is already done):
1. [done] `python -m perinv.train.tune --algos sac,ppo,a2c,maskppo,dqn --envs configs/env/n{1,2,5}_k{06,10}.yaml --jobs 8`
   → `configs/tuned/<algo>-<env>.yaml` + `results/tune/*.csv`. Commit the tuned files.
2. Tag the frozen env: `git tag v0.1-frozen-env` (env, SPEC, budget, tuned hparams). Any later env change = rerun.
3. `python scripts/run_matrix.py configs/experiments/core.yaml --jobs 8 --official` (clean tree required).
4. `python scripts/analyze.py` → `results/` tables and figures for the report.
5. Optional, only if time remains: `calibrated.yaml` (group D), `shocks.yaml`.

Open team decisions (DECISIONS.md "team to confirm"): D1–D8, the 500k budget, the N=5 capacity 80/48, the M2
choices (shared 64×64 net, SAC train_freq = 1), and whether to run the optional groups at all.

Repo admin: CODEOWNERS handles are placeholders; fork PRs run no CI (add teammates as collaborators instead).
CI does run on Python 3.15 for pushes (green).

## M1 checklist
Must have:
1. [x] U1–U3: dynamics, env + wrappers, exact DP (all M1 pending guards replaced by real tests).
2. [x] **O_ref → C per κ** applied (provisional stand-in; `configs/env/oref_provisional.json`). Remeasure with M2's (s,S):
   `measure_oref.py ... --json`, `make_calibrated_configs.py ... --oref`, then edit the six core configs.
3. [x] **Exact DP** (no truncation: the d ≥ I row weighted by P(D ≥ I), lost sales −b·E[(D−I)⁺]); equals brute force to 1e-9 and the simulated policy within 4 SE.
4. [x] **DP with seasonality and fractional λ** (calibrated N=1 configs solve; simulation agrees).

Should have:
5. [x] **Cold-start effect** measured (`dp.backward_induction.daily_profile`): empty start costs ≈ 15–18 % of V0; warm start (D4) applied.
6. [x] **Benchmarks** (`scripts/bench_vecenv.py`, Windows): PPO ≈ 2.9k env steps/s with DummyVecEnv×4 (1M steps ≈ 6 min); rerun on Linux.

Optional:
7. [ ] BSP-low-EW baseline (Haijema & Minner 2019), a strong age-aware reference for N ≥ 2.
8. [ ] Cross-check the DP against mdpax (Farrington et al.) on a small instance.

Done by decision: the mapping endpoints (D1) and the clipping rule (D2).

## What exists
| Status | What |
|---|---|
| Working + tested | `env/` (config, dynamics, env, wrappers), `dp/` + `baselines/dp_policy.py` (exact DP, cached), `data/` (CRN, calibration, scenarios), `eval/` (evaluate, writer, warehouse, metrics, stats), `runner.py`, `utils/`, `baselines/base.py`; scripts `make_crn_paths`, `solve_dp`, `measure_oref` (provisional), `bench_vecenv`, `make_calibrated_configs`, `make_shock_configs`, `run_matrix`, `analyze`; infra data platform (`infra/`) |
| Stub (docstring only) | none for the core experiments |
| Provisional values | cost parameters (SPEC §1); capacities confirmed with the real (s,S) on 2026-11-23 |
| Tests | 193 passed, plus 2 slow tests (`pytest -m slow`) |

Platform notes:
- Final experiments run on Linux or WSL2. Development also works on native Windows (docs/infra/RUN.md).
- On Windows without MSVC, install with `PYYAML_FORCE_LIBYAML=0` (there is no cp315 wheel).
- SubprocVecEnv and the runner use spawn: always run real script files with an `if __name__ == "__main__":` guard.

## History
- 2026-11-23 (Weibo): **Tuning done** at 500k steps, 10 trials, 22 algorithm × cell pairs, 8 jobs, 2 h 40 m on an M5
  Mac; no failures. Best tune-path profit (SAC / PPO / A2C): n1_k06 506.2 / 505.7 / 480.4, n1_k10 667.0 / 662.8 /
  624.5, n2_k06 972.1 / 964.4 / 918.6, n2_k10 1340.6 / 1319.8 / 1246.3, n5_k06 2472.6 / 2446.4 / 2328.6, n5_k10
  3322.8 / 3258.0 / 3042.1; MaskablePPO 506.3 / 660.8, DQN 506.4 / 662.6 (N=1). Defaults stayed best for A2C in 4
  of 6 cells. A2C's best and PPO's default on n5_k06 are both exactly 2328.626 (likely the same fill-capacity
  policy); check in the analysis. Per-trial tables in results/tune/ (gitignored).
- 2026-11-23 (Weibo): **M2 implemented**: `agents/factory.py`, `agents/sb3_policy.py`, `train/{train,callbacks,tune}.py`,
  `baselines/{s_S,newsvendor}.py`, `scripts/smoke_e2e.py`; algo defaults and Optuna spaces filled; M2 guard tests
  real (`tests/test_agents_baselines.py`). **Review fixes**: D8 terminated at T; paired_vs native-only baseline;
  pairing on crn_test_sha256; builder/env factory/tune sha in train_key; DP_VERSION. Smoke run: PPO 10k steps =
  94.3 % of DP on n1_k06. Core matrix at the tiny profile: 36/36 done. N=5 capacity 80/48. Budget 500k. Tests:
  193 fast + 2 slow pass.
- 2026-11-22 (M3): Text writers emit LF on every OS (CRN manifest, run files, log lines). Performance and compute projection (EXPERIMENTS.md "Compute budget"): training dominates, and SAC with SB3 defaults is about 90 % of all CPU time (0.11k steps/s versus 2.8k for PPO); parallel jobs scale ×8.5 at 16. Env, eval, warehouse and runner overheads are small; no optimization applied yet.
- 2026-11-22 (M1): U5 applied the U4 decisions. D4 warm start (`dynamics.initial_state`, `EVENT_ORDER_VERSION = "2"`; env, DP and test double share it). O_ref remeasured with the warm start and saved to `configs/env/oref_provisional.json`; core capacities 10/16, 19/32, 47/79; calibrated configs regenerated with `--oref` (m3 world, a demand_hash guard rejects stale measurements): 8/13, 16/27, 22/36. Shock configs regenerated. CRN paths committed (`data/crn`, 19 sets, unaffected by capacity). DP V0 now: n1_k06 505.7, n1_k10 661.3, frn m1/m3 N=1 κ0.6 236.5/302.7.
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
