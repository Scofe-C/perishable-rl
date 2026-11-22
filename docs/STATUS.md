# Project status

> Keep this current. Whoever finishes a step updates "Current step" and adds one line to "History".
> Agents: read this first, then [SPEC.md](SPEC.md). Project summary: [README.md](../README.md).

**Last updated:** 2026-11-22 · **Phase:** Week 1. The M3 pipeline is done. The M1 env and DP start now on
branch `zhengxin-m1-env` (Zhengxin covers M1 as well as M3).

## Current step
**Build the M1 core, then the thinnest end-to-end run.** Units, each reviewed before the next:
~~U1 `env/dynamics.py`~~ (done) → **U2 env + wrappers** → U3 DP.

Blockers:
1. **SPEC.md is a draft.** D1–D6 (DECISIONS.md) are written in and need the other two members' confirmation.
2. **No env, training or baselines yet**, so every runner job shows `blocked (no builder yet)`. The pipeline
   is tested end to end only with the `tests/fakes.py` test double.
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
1. [ ] U1–U3: dynamics (done), env, wrappers and DP, replacing the M1 skips in `tests/test_pending_guards.py`.
2. [ ] **O_ref → C per κ** (`scripts/measure_oref.py`). Use a provisional order-up-to stand-in until M2's (s,S)
   exists, then remeasure. Record the result in DECISIONS.md. Until then capacity = `κ·20·N`.
3. [ ] **Exact DP.** Demand d ≥ I (on hand after arrival) all lead to the same next state, so that row is weighted
   by P(D ≥ I), and the lost-sales term is added analytically as −b·E[(D−I)⁺]. No truncation is needed.
   Acceptance: `V[0, s0]` agrees with the DP policy's mean profit on the 100 test paths, within the CI.
4. [ ] **DP with seasonality and fractional λ:** the calibrated N=1 configs (groups DP1/DP3/DP13).

Should have:
5. [ ] **Cold-start effect** (D4): compare V from an empty start with V from a steady-state start.
6. [ ] **Benchmarks:** env step time (feeds `bench_vecenv.py` and `train_env_steps`) and DP runtime.

Optional:
7. [ ] BSP-low-EW baseline (Haijema & Minner 2019), a strong age-aware reference for N ≥ 2.
8. [ ] Cross-check the DP against mdpax (Farrington et al.) on a small instance.

Done by decision: the mapping endpoints (D1) and the clipping rule (D2).

## What exists
| Status | What |
|---|---|
| Working + tested | `env/config.py`, `env/dynamics.py`, `data/` (CRN, calibration, scenarios), `eval/` (evaluate, writer, warehouse, metrics, stats), `runner.py`, `utils/`, `baselines/base.py`; scripts `make_crn_paths`, `make_calibrated_configs`, `make_shock_configs`, `run_matrix`, `analyze`; infra data platform (`infra/`) |
| Stub (docstring only) | `env/perishable_env.py`, `env/wrappers.py`, `dp/`, `baselines/{dp_policy,s_S,newsvendor}.py`, `agents/`, `train/`; scripts `smoke_e2e`, `bench_vecenv`, `solve_dp` |
| Provisional values | capacities (`κ·20·N`), cost parameters (SPEC §1) |
| Tests | 126 passed, 7 skipped (the pending pitfall guards), plus 1 slow test (`pytest -m slow`) |

Platform notes:
- Final experiments run on Linux or WSL2. Development also works on native Windows (docs/infra/RUN.md).
- On Windows without MSVC, install with `PYYAML_FORCE_LIBYAML=0` (there is no cp315 wheel).
- SubprocVecEnv and the runner use spawn: always run real script files with an `if __name__ == "__main__":` guard.

## History
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
