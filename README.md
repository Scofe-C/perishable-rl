# perishable-rl

This project compares PPO, A2C, SAC, DQN and MaskablePPO (Stable-Baselines3 / sb3-contrib, no rewritten
algorithms) on a perishable, multi-SKU inventory problem with shared storage capacity. Results are measured
against exact DP (N=1) and tuned heuristics (N≥2), on the same CRN demand paths and under the same budgets.

**Docs:** [STATUS](docs/STATUS.md) (where we are, what's next) · [SPEC](docs/SPEC.md) (contracts: event order,
action mappings, CRN format) · [EXPERIMENTS](docs/EXPERIMENTS.md) (matrix, fairness rules) ·
[DECISIONS](docs/DECISIONS.md) (what changed and why) · [WAREHOUSE](docs/WAREHOUSE.md) (result files, keys,
metrics) · [infra/RUN](docs/infra/RUN.md) (commands) · [infra/STRUCTURE](docs/infra/STRUCTURE.md) (data flow).

## Setup
**Python 3.15** (pins resolved on 3.15.2). CPU only.
```bash
python3.15 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -e .
pytest -q
```
Final experiments run on Linux or WSL2 (`bash scripts/setup_linux_vm.sh` for a fresh VM). For development on
native Windows, see docs/infra/RUN.md.

## Layout and owners
| Path | What | Owner |
|---|---|---|
| `src/perinv/env/` | dynamics, Gymnasium env, action mappings | M1 (all 3 review) |
| `src/perinv/dp/`, `baselines/dp_policy.py` | exact DP for N=1 | M1 |
| `src/perinv/baselines/` | (s,S), newsvendor | M2 |
| `src/perinv/agents/`, `train/` | SB3 model factory, training, Optuna tuning | M2 (PPO, A2C), M3 (SAC, DQN, MaskablePPO) |
| `src/perinv/data/` | CRN demand paths, real-data calibration, demand-shock scenarios | M3 |
| `src/perinv/eval/`, `runner.py` | the single evaluation path, result warehouse, metrics, matrix runner | M3 |
| `infra/` | FreshRetailNet data platform (Docker, dbt, Dagster) | M3 |
| `configs/` | env cells (N × κ), experiment matrices, algo defaults, budgets | all |

## Rules
- `docs/SPEC.md`, `src/perinv/env/` and `configs/budget.yaml` change only with all three members (DECISIONS.md).
- Every agent is scored by `perinv.eval.evaluate` on the `test` CRN paths; tuning uses `tune` paths only.
- Every entry point calls `perinv.utils.seeding.init_process(seed)` inside `if __name__ == "__main__":`.
- Experiments run through `scripts/run_matrix.py`, which writes `runs/<group>-N<N>-k<κ>-<policy>-s<seed>/`.
