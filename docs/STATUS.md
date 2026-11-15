# Project status

> Keep this file current. Whoever finishes a step updates "Current step" and adds a line to "History".
> Agents: read this first, then [SPEC.md](SPEC.md).

**Last updated:** 2026-11-13 · **Phase:** Week 1, Day 1 (skeleton and Python 3.15 env done; contracts not yet ratified)

## Where this project comes from
The source is the team draft *"Perishable Inventory RL Project: Datasets, Methods and Experiment Plan"*
(Nov 12, 2026). In short:
- We compare PPO, A2C, SAC (Box actions) and DQN, MaskablePPO (discrete, N=1 only) using
  Stable-Baselines3 / sb3-contrib. We do not rewrite algorithms.
- One shared Gymnasium env: perishable stock tracked by age bucket, FIFO issuing, fixed lead time τ,
  Poisson demand, lost sales, and a shared capacity C across SKUs.
- References: exact DP for N=1 ("% of optimal"); (s,S) and newsvendor for N≥2.
- Fair comparison: the same env-step budget, 10 tuning trials per algorithm, the same 100 CRN test
  paths with paired per-path differences, and IQM with bootstrap CIs (via scipy; rliable was dropped, see DECISIONS.md).
- About 110 runs, CPU only. The timeline is 5 weeks.

Those decisions now live in [SPEC.md](SPEC.md) (contracts), [EXPERIMENTS.md](EXPERIMENTS.md)
(matrix and rules) and [DECISIONS.md](DECISIONS.md) (changes).

## Current step
**Week 1: ratify contracts, then build the thinnest end-to-end run.**

Done: **Python 3.15 environment and pinned requirements.** Every teammate should install Python 3.15
(python.org installer on macOS, deadsnakes on Ubuntu, or `scripts/setup_linux_vm.sh`), then run
`python3.15 -m venv .venv && pip install -r requirements.txt -r requirements-dev.txt -e .`.

Blockers to clear first:
1. **SPEC.md is a DRAFT.** The event order (§3), action mappings (§4) and default parameters (§1)
   need agreement from all three members before anyone implements the env.
2. **Confirm two scaffold decisions** in DECISIONS.md: Python 3.15, and dropping rliable for scipy-based IQM/CI.
3. **Repo admin.** Nothing is committed yet: the files are staged, there's no git remote, and the CODEOWNERS handles are placeholders.
   CI on Python 3.15 has not run yet because there's no remote.

Then, in parallel (see README owner table):
- **M1:** `env/dynamics.py`, `env/perishable_env.py`, `env/wrappers.py` and their tests. Next come the DP and the conformance test.
- **M2:** `agents/factory.py`, `agents/sb3_policy.py`, `train/train.py` and `callbacks.py`, and the newsvendor baseline.
- **M3:** `eval/evaluate.py` and `metrics.py`. Generate and commit CRN paths with `python scripts/make_crn_paths.py configs/env/*.yaml`.
- **All, by Friday:** `scripts/smoke_e2e.py` (N=1 → PPO 10k steps → 5 paths → plot), `bench_vecenv.py`,
  then set `train_env_steps` in `configs/budget.yaml`.

## What exists vs. what is a stub
| Status | Files |
|---|---|
| Working + tested | `env/config.py` (EnvConfig, YAML loader, hashes), `data/crn.py` (generate/write/load with manifest sha256 check), `scripts/make_crn_paths.py`, `utils/seeding.py`, `utils/run_io.py`, `baselines/base.py` (Policy protocol) |
| Docstring stub only | everything else in `src/perinv/` and `scripts/`; each docstring names its owner and contract |
| Provisional values | `configs/env/*.yaml` capacity = round(κ·20·N) until O_ref is measured; cost params in SPEC §1 |
| Tests | `tests/test_config_and_crn.py` (9 pass); `tests/test_pending_guards.py` (8 skipped TODOs, one per pitfall: replace each with a real test) |

Last check (2026-11-13, Python 3.15.2, macOS arm64):
- `pip check` is clean, `ruff check .` is clean, and `pytest -q` gives 9 passed, 8 skipped.
- A fresh venv installed from the pins matches the dev venv exactly.
- Short training runs of PPO, SAC, DQN and A2C (with SubprocVecEnv) work on toy Gymnasium envs, and MaskablePPO imports.
- A scipy IQM bootstrap CI works.
- Linux x86_64 cp315 wheels exist for every pin, including torch 2.15.0+cpu. PyYAML builds from its pure-Python sdist.

Gotcha: SubprocVecEnv fails if the script is fed through stdin or lacks the `__main__` guard (macOS/3.15
use forkserver/spawn). Always run real script files.

## History
- 2026-11-14 (M3): Added cross-evaluation to the runner. `eval_envs` in a matrix group builds on A and scores on B (demand may differ, nothing else), and reuses A's trained model. The warehouse pairs on the eval world; `metrics.paired_groups` gives the Q4 calibration cost; group D now has D13 and DP13. Also fixed a Windows-only race: a parallel worker reading a manifest made another worker's final `os.replace` fail, which left that run stuck at `running`.
- 2026-11-14 (M3): Real-data calibration:
  - `perinv/data/calibration.py` and `scripts/make_calibrated_configs.py` turn the infra C1 file into `configs/env/calibrated/*.yaml` (12 files: m1/m3 × N 1,2,5 × κ 0.6,1.0).
  - `configs/experiments/calibrated.yaml` defines group D.
  - Docs for M3 infra: `docs/infra/STRUCTURE.md` and `docs/infra/RUN.md`.
  - Open item: cross-evaluation (build on m1, evaluate on m3) is not in the runner yet.
- 2026-11-14 (M3): Warehouse reader and analysis done. Phase 1 infra is complete.
  - `eval/warehouse.py`: `connect()` sees done runs only and uses fixed types.
  - `eval/sql/views.sql`: the views `v_step`, `v_path`, `v_day`, `v_run` and `v_vs_dp`. Every metric is defined here once.
  - `eval/metrics.py` and `eval/stats.py`: IQM, percentile bootstrap and paired differences.
  - `scripts/analyze.py`: writes `runs.csv`, `summary.csv` and `profit_<group>.png`.

  Result: 50 passed, 8 skipped, plus 1 slow test.
- 2026-11-14 (M3): Runner done: `src/perinv/runner.py`, `scripts/run_matrix.py` and `configs/experiments/core.yaml` (124 jobs).
  - Skips finished runs, runs K jobs in parallel (spawn), and a failure in one job does not stop the others.
  - Resumes after a hard kill. This is tested with a real subprocess (`-m slow`).
  - Builders are plugged in as "module:function" (WAREHOUSE.md §8). Every job is "blocked" until `train.build` and the baseline builders exist.

  Result: 38 passed, 8 skipped, plus 1 slow test.
- 2026-11-14 (M3): Rewrote `utils/run_io.py` per WAREHOUSE.md §4:
  - `train_key` / `run_key` (no git SHA)
  - `run_context` (running → done/failed, `failure.json`, `log.jsonl`)
  - `run_status`
  - `official` mode refuses a dirty tree
  - package list via `importlib.metadata`. The old `pip freeze` silently returned nothing in pip-less (uv) venvs.

  `tests/test_run_io.py` added. Result: 30 passed, 8 skipped.
- 2026-11-14 (M3): `eval/evaluate.py`, `eval/writer.py` and `utils/atomic.py` are implemented against docs/WAREHOUSE.md and tested with a SPEC §3 test double (`tests/fakes.py`). Result: 19 passed, 8 skipped. A file for 100 paths × 30 days is 2–5 bytes/row and takes about 10 ms to write.
- 2026-11-14 (M3): Added duckdb 2.0.0 (see DECISIONS.md). Verified the pinned env on native Windows with Python 3.15.2: 9 passed, 8 skipped, ruff clean. PyYAML has no Windows cp315 wheel, so install it with `PYYAML_FORCE_LIBYAML=0` to skip the C extension when no MSVC is present.
- 2026-11-13: The plan was turned into a repo skeleton. Docker was dropped because nothing is
  deployed; pinned requirements plus `scripts/setup_linux_vm.sh` cover Linux runs. CRN paths
  generated during the smoke check were deleted so they can be regenerated with the pinned env.
- 2026-11-13: Homebrew 3.12 was broken, so we moved to Python 3.15.2. Requirements are pinned
  (36 runtime packages, no pre-releases), and rliable was dropped because it is incompatible with
  pandas 3. Install, tests and SB3 training were verified. CRN paths are still to be generated.
