# Experiments

## Matrix (core: about 110 runs)
| Group | N | κ | Algorithms | Seeds | Runs | Reference |
|---|---|---|---|---|---|---|
| A | 1 | 0.6, 1.0 | PPO, A2C, SAC, DQN, MaskablePPO | 5 | 50 | DP |
| B | 2, 5 | 0.6, 1.0 | PPO, A2C, SAC | 5 | 60 | (s,S), newsvendor |

If compute runs short, first cut seeds from 5 to 3 (`run_matrix.py --seeds 0,1,2`), then drop MaskablePPO.

## Fair-comparison rules (fixed before running)
1. **Same training budget.** Every algorithm gets the same env-step budget (`configs/budget.yaml`). Log the actual steps, because SB3's `total_timesteps` is only a lower bound.
2. **Same tuning budget.** Exactly 10 Optuna trials per algorithm × cell, scored on the `tune` CRN paths only.
3. **Same evaluation paths.** The 100 `test` CRN paths, with differences computed pairwise per path.
4. **Same action mapping.** PPO, A2C and SAC share `CapacityShareAction`.
5. **Statistics.** IQM (25% trimmed mean) with bootstrap CIs over seeds, via `scipy.stats` (rliable is not installable on Python 3.15, see DECISIONS.md).

## Metrics
Mean profit (with CI) · % of DP (N=1; target ≥ 95%) · waste rate = expired / ordered · fill rate = sold / demand ·
capacity utilisation = mean occupancy / C · learning curve (profit vs actual steps) · CPU-hours per run.
