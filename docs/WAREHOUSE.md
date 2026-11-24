# WAREHOUSE — evaluation results contract

> The code is the source of truth: the pyarrow schema in `perinv.eval.writer` and the SQL in
> `perinv/eval/sql/views.sql`. Any column change bumps `EVAL_SCHEMA_VERSION` and gets a DECISIONS.md entry.

## 1. What this is
Every evaluated agent (SB3 models, DP, (s,S), newsvendor) gets one per-step result file. All analysis reads
those files with SQL (DuckDB, no server).
- **One writer:** only `perinv.eval.evaluate` + `writer` produce result files.
- **One reader:** `perinv.eval.warehouse.connect()` returns a DuckDB connection with the views in §5. Do not glob files yourself.
- **Scope:** final evaluations on the `test` split only. Tuning trials, learning curves and checkpoints stay as files in the run folder.

## 2. Layout
```
runs/<run_name>/                 # run_name = <group>-N<N>-k<κ>-<policy>-s<seed>; runs/ is gitignored
  manifest.json                  # lineage + status (§4) = the run dimension
  config.yaml, log.jsonl         # resolved config; one JSON line per event (carries the stage)
  failure.json                   # only on failure
  model.zip, vecnormalize.pkl    # RL training outputs, if any
  eval_test.parquet              # fact_step for this run (§3)
```
No hive partitioning: the core matrix is about 0.8M rows (~10 MB), and folder partitions made DuckDB read
`kappa=0.6` as text (`WHERE kappa = 1` then matched it). Filters join `dim_run` instead.

## 3. `fact_step`: one row per (run, path, day, SKU)
Unique grain `(run_key, path_id, t, sku)`, sorted by `path_id, t, sku`, zstd-compressed.

| Column | Type | Meaning (SPEC §3 steps) |
|---|---|---|
| `run_key` | string (dictionary) | idempotency key (§4), same on every row |
| `path_id`, `t`, `sku` | int16 | test path index k, day 0…T−1, SKU 0…N−1 |
| `occ` | int32 | this SKU's occupancy at decision time (before ordering) |
| `order` | int32 | units that entered the pipeline (after clipping) |
| `sold`, `lost` | int32 | step 3; demand = sold + lost |
| `expired` | int32 | step 4 |
| `held` | int32 | step 5, on hand after expiry |
| `day_reward` | float64, **sku = 0 row only** | the env's reward for the day; NULL on other rows |
| `clipped` | bool, **sku = 0 row only** | the env clipped today's order; NULL on other rows |

- **Quantities only.** Money is quantity × the per-SKU prices in `dim_run`, computed in the views.
- **Day values on the sku = 0 row only**, so `SUM(day_reward)` cannot count a day N times.
- **`day_reward` is kept to reconcile** against the reward recomputed from quantities (§6).

File metadata: `perinv.eval_schema_version`, `perinv.run_key`, `perinv.n_paths`/`T`/`N`.

## 4. Run identity, lineage and status (`manifest.json` = `dim_run`)

### 4.1 Keys
| Key | Built from | Used for |
|---|---|---|
| `train_key` | build config's `config_hash`, policy + hash of resolved hparams, seed, `train_env_steps`, `EVENT_ORDER_VERSION` | reuse a trained model (skip training if it matches) |
| `run_key` | `train_key` + eval world's `config_hash`, `demand_hash`, sha256 of `test.npz`, `EVAL_SCHEMA_VERSION` | "is this evaluation done?"; key on every fact row |

Git SHA, host, timestamps and library versions are recorded in the manifest but are **not** in the keys:
a key holds only what changes the result, otherwise every docs commit would rerun the matrix.

### 4.2 Cross-evaluation and demand shocks
A matrix group may pair `envs` (build world A) with `eval_envs` (score world B). A and B may differ only in
`demand_mean`, `season` and `scenario`; a scenario config is only allowed as B, and must keep A's base demand.
`train_key` comes from A, so the model is reused; the keys and the file describe B. The manifest adds
`eval_env_path`, `eval_config_hash`, `eval_demand_hash`, `eval_demand_mean` and `cross_eval`.

### 4.3 Required manifest fields
`schema_version, run_name, run_key, train_key, status, group, policy, seed, n_skus, kappa, capacity, horizon,
env` (full EnvConfig with per-SKU cost lists), `config_hash, demand_hash, crn_test_sha256, train_env_steps,
actual_env_steps, git_sha, git_dirty, python, pip_freeze, host, start_time, end_time, cpu_seconds,
wall_seconds, artifacts`. On failure also `error = {stage, type, message}`, mirrored in `failure.json`.

### 4.4 Status and "done"
`running` → `done` | `failed`. **Done** requires all three: manifest status `done`; manifest `run_key` equals the
key from the current config; `eval_test.parquet` exists with the same `run_key` in its metadata.
The eval file is written (tmp + rename) before the manifest says `done`, so a crash at any point leaves
"not done" and the next run redoes it. A changed config makes the run stale, and it is rerun. A worker that dies
without unwinding is marked `failed` by the runner (`WorkerCrashed`, last logged stage).

## 5. Views (`perinv.eval.warehouse`, SQL in `eval/sql/views.sql`)
Only done runs appear.

| View | Grain | Adds |
|---|---|---|
| `dim_run` | run | manifest fields |
| `fact_step` | run × path × t × sku | raw rows |
| `v_step` | run × path × t × sku | demand; revenue/ordering/holding/expiry/lost_sales and `sku_reward` from that SKU's prices |
| `v_path` | run × path | `profit = SUM(day_reward)`; totals ordered/sold/lost/expired/demand |
| `v_day` | run × path × t | total `occ`, `occ + order` |
| `v_run` | run | the metrics below, with group, N, κ, policy, seed |
| `v_vs_dp` | run | % of DP and gap to DP |

Metrics (pooled over all paths, days and SKUs of a run; ratios of sums, not means of ratios):
- **Mean profit** = mean over paths of `v_path.profit`.
- **Waste rate** = `SUM(expired) / SUM(order)`; **fill rate** = `SUM(sold) / SUM(sold + lost)` (NULL when there is no demand).
- **Capacity utilisation** = mean over (path, t) of `SUM_sku(occ) / capacity` (decision time).
- **% of DP** (N=1) = `mean(profit) / mean(profit_dp)` on the same paths and eval world, against the DP run
  natively in that world. NULL if the DP's mean profit ≤ 0 (use `mean_gap_to_dp`), and NULL in shocked
  worlds, which have no native DP (the DP is not optimal under shocks).
- IQM and bootstrap CIs over seeds: `perinv.eval.stats`, applied by `perinv.eval.metrics`.

## 6. Checks
`evaluate` rejects (and the run fails):
- actions that are not N whole, finite, non-negative numbers (orders above R are legal; the env clips);
- `info` quantities that are not whole units;
- episodes that end before T or are terminated.

The writer checks the following before writing anything:
1. The shapes match the config, and the grain is unique.
2. All quantities are ≥ 0, and `sold + lost` equals the CRN demand.
3. Every day, `SUM_sku(occ + order) ≤ capacity`.
4. Every reward is finite, and `|SUM_sku(sku_reward) − day_reward| ≤ 1e-6`.

The reader refuses files whose schema version or `run_key` disagrees with the manifest.

## 7. Builder contract (M1 DP, M2 training and baselines)
The runner calls `build(ctx) -> Policy` at `perinv.runner.DEFAULT_BUILDERS`: `perinv.train.train:build` (RL)
and `perinv.baselines.{dp_policy,s_S,newsvendor}:build`. `ctx` carries `cfg, policy, hparams, seed,
train_env_steps, run, make_env, reuse_model, model_dir, crn_root`. Build from `ctx.cfg` only, never from the eval world.

An RL builder:
1. If `ctx.reuse_model` is true, it loads `model.zip` / `vecnormalize.pkl` from `ctx.model_dir`, which may be another run's folder.
2. Otherwise, it trains and saves them into `ctx.run.dir`.
3. It calls `ctx.run.update(train={"train_key": ctx.run.train_key, "status": "done"}, actual_env_steps=...)`.
4. It returns a Policy with frozen VecNormalize that outputs order quantities (SPEC §8).

The runner does the seeding, the manifest, the evaluation and the writing.
