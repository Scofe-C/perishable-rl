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

## Extension groups (outside the core matrix; core results never depend on them)
| Matrix | Groups | Question | Compare with |
|---|---|---|---|
| `configs/experiments/calibrated.yaml` | D1, D3, D13, DP1/DP3/DP13 | What does a policy lose when its demand was calibrated from sales that ignore stockouts? | `metrics.paired_groups(con, "D13", "D3")` |
| `configs/experiments/shocks.yaml` | A_*/B_* (spike, drop, outage, shift, noisy) | What does a policy trained on base demand lose under demand shocks? (eval only) | `metrics.paired_worlds(con, "A_spike", "A")` |

## Compute budget (measured 2026-11-22, Windows, i9-13900HX 24 cores, 16 GB; rerun on the Linux box)
Training dominates everything else. One evaluation run takes about 0.2 s (+0.3 s of SB3 inference); the DP needs
0.1 s; the warehouse handles 480 runs / 3.8M rows (2.5 B/row) in under a second.

Throughput with SB3 defaults, one torch thread (env steps/s): PPO 2.8k (DummyVecEnv×4), A2C 4.6k, DQN 2.5k,
MaskablePPO 2.6k, **SAC 0.11k**. SAC variants: train_freq 4 gives 0.41k; net 64×64 gives 0.29k; net 64×64 with
batch 64 gives 0.34k; 4 envs give 0.45k. Parallel PPO jobs: aggregate speed-up ×6.5 at 8 jobs and ×8.5 at 16
(8 P-cores, then E-cores).

Runs, including 10 tuning trials per algorithm × cell at the full budget: PPO 100 (with group D), A2C 90, SAC 90,
DQN 30, MaskablePPO 30. The shock and cross-eval groups reuse trained models (eval only).

| train_env_steps | CPU-h, SB3 default SAC (SAC share) | wall at 16 jobs | CPU-h, SAC train_freq 4 | wall |
|---|---|---|---|---|
| 200k | 48 (91 %) | ≈ 6 h | 17 | ≈ 2 h |
| 500k | 120 (91 %) | ≈ 14 h | 41 | ≈ 5 h |
| 1M | 239 (91 %) | ≈ 28 h | 83 | ≈ 10 h |
