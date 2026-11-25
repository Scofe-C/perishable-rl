# Core experiment results (v0.1-frozen-env)

All 124 official runs of `configs/experiments/core.yaml`: group A (N = 1) and group B (N = 2, 5), κ = 0.6 and 1.0,
5 seeds per RL algorithm, 500k env steps each, tuned hyperparameters from `configs/tuned/`. Code at commit
`ed56d2c` (tag `v0.1-frozen-env`), clean tree (`official = True` in every manifest). 12.2 CPU-hours on an M5 Mac.

Regenerate from `runs/` with `python scripts/analyze.py`; copied here because `results/` is gitignored.
Every profit below is on the 100 **test** CRN paths, shared by all policies. "IQM" = interquartile mean over seeds,
CIs are 95 % percentile bootstrap.

## Files
| File | Content |
|---|---|
| `q1_pct_of_dp.csv` | N = 1: profit as % of the exact DP (paired on the same paths), IQM over seeds |
| `q2_vs_best_heuristic.csv`, `.png` | every cell: per-path profit difference to the stronger of (s,S) and newsvendor |
| `q3_efficiency.csv`, `q3_curves_N{1,2,5}.png` | CPU hours per run, env steps to 95 % of the final tune profit, learning curves (tune paths) |
| `summary.csv`, `profit_{A,B}.png` | IQM + CI of every metric per (group, N, κ, policy) |
| `runs.csv` | one row per run |
| `tune/*.csv` | every Optuna trial (10 per algorithm × cell) |
| `core_official*.log` | runner logs (the rerun covers 6 jobs refused by the dirty-tree check, see STATUS History) |

## Results
**Q1, N = 1, % of DP:** SAC 99.9 / 99.5 (κ = 0.6 / 1.0), PPO 99.9 / 99.1, MaskablePPO 99.8 / 99.0, DQN 99.9 / 98.7,
A2C = (s,S) 95.2 / 94.0, newsvendor 95.2 / 85.3.

**Q2, % of the best heuristic** (κ = 0.6 / 1.0):

| N | SAC | PPO | A2C |
|---|---|---|---|
| 1 | 104.9 / 105.8 | 104.9 / 105.3 | 100.0 / 100.0 |
| 2 | 105.0 / 105.9 | 104.5 / 102.9 | 99.4 / 97.9 |
| 5 | 104.9 / 105.1 | 103.8 / 102.8 | 98.6 / 96.3 |

SAC and PPO beat the best heuristic on all 5 seeds in every cell (paired CI excludes 0). SAC holds its margin as N
grows; PPO's shrinks at κ = 1.0; A2C falls below the heuristic.

**Q3, cost:** SAC 0.26–0.39 CPU-h per run (≈ 400–570 env steps per CPU second), PPO 0.02–0.05, A2C 0.02. SAC reaches
95 % of its final tune profit in 25k–150k steps, PPO in 75k–450k.

## Caveats for the report
- **A2C collapses** to one deterministic policy in every seed (zero-width CI); on N = 1 it is the (s,S) policy
  (same profit). This is also why A2C's best and PPO's default tune score on n5_k06 are both 2328.626.
- **κ = 0.6, N = 1 is easy:** filling capacity already gives 95.2 % of DP, so % of DP there says little about learning.
- **DQN relies on the env's clipping:** it asks for more than the remaining capacity on up to 88 % of days and the
  env clips it (D2, allowed by SPEC). Box agents and MaskablePPO never clip.
- Wall times in the manifests include about 9 h of laptop sleep; use `cpu_hours`, not wall time.
