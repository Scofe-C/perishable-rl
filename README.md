# perishable-rl

This project compares PPO, A2C, SAC, DQN and MaskablePPO on a perishable, multi-SKU inventory problem
with shared storage capacity. Results are measured against exact DP (N=1) and tuned heuristics (N≥2).

**Read first:** [docs/SPEC.md](docs/SPEC.md) (the frozen contracts), [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
(the matrix and fairness rules) and [docs/DECISIONS.md](docs/DECISIONS.md) (what changed and why).

## Setup
macOS or Linux, **Python 3.15** (the pins in requirements.txt were resolved on 3.15.2). On Windows use WSL2, because Gymnasium does not officially support Windows.
```bash
python3.15 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -e .
pytest -q
```
For a fresh cloud Linux VM, run `bash scripts/setup_linux_vm.sh`. Everything runs on CPU, so no GPU is needed.

## Layout and owners
| Path | What | Owner |
|---|---|---|
| `src/perinv/env/` | dynamics, Gymnasium env, action mappings | M1 (all 3 review) |
| `src/perinv/dp/`, `baselines/dp_policy.py` | exact DP for N=1 | M1 |
| `src/perinv/baselines/` | (s,S), newsvendor | M2 |
| `src/perinv/agents/`, `train/` | SB3 model factory, training, Optuna tuning | M2 (PPO, A2C), M3 (SAC, DQN, MaskablePPO) |
| `src/perinv/data/` | CRN demand paths | M3 |
| `src/perinv/eval/` | the single evaluation path, metrics, IQM + bootstrap CIs (scipy) | M3 |
| `configs/` | env cells (N × κ), algo defaults, tuning spaces, budgets | all |
| `scripts/` | CLI entry points | see each file |

## Rules
- Every entry point calls `perinv.utils.seeding.init_process(seed)` inside `if __name__ == "__main__":`.
- Every agent is evaluated through `perinv.eval.evaluate` on the `test` CRN paths. There are no ad-hoc eval loops.
- Run outputs go to `runs/<group>-N<N>-k<κ>-<algo>-s<seed>/` with a manifest (git SHA, pip freeze, CPU time).
- Changes to `src/perinv/env/`, `docs/SPEC.md` or `configs/budget.yaml` need all three approvals (CODEOWNERS).

## Week-1 checklist
- [ ] Ratify SPEC.md and fill in the CODEOWNERS handles
- [ ] `python scripts/make_crn_paths.py configs/env/*.yaml` and commit `data/crn/`
- [ ] Env + action mapping + their tests (M1); evaluate.py (M3); train.py + sb3_policy.py (M2)
- [ ] `python scripts/smoke_e2e.py`: N=1 → PPO 10k steps → 5 paths → plot
- [ ] `python scripts/bench_vecenv.py`, then time one run and set `train_env_steps` in `configs/budget.yaml`
- [ ] Tag `v0.1-frozen-env` once the env and conformance tests pass
