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
- 2026-11-13: The plan was turned into a repo skeleton. Docker was dropped because nothing is
  deployed; pinned requirements plus `scripts/setup_linux_vm.sh` cover Linux runs. CRN paths
  generated during the smoke check were deleted so they can be regenerated with the pinned env.
- 2026-11-13: Homebrew 3.12 was broken, so we moved to Python 3.15.2. Requirements are pinned
  (36 runtime packages, no pre-releases), and rliable was dropped because it is incompatible with
  pandas 3. Install, tests and SB3 training were verified. CRN paths are still to be generated.
