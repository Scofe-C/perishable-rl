# WAREHOUSE — evaluation results contract

> **Status: DRAFT (M3, 2026-11-14), for team review.** The code is the source of truth once it exists:
> the pyarrow schema in `perinv.eval.writer` and the views in `perinv.eval.warehouse`. Any column change bumps
> `EVAL_SCHEMA_VERSION` and gets a line in DECISIONS.md.

## 1. What this is
Every evaluated agent (SB3 models, DP, (s,S), newsvendor) gets one per-step result file. All analysis
reads those files with SQL (DuckDB), so there is no database server.

- **One writer.** Only `perinv.eval.evaluate` writes result files (CLAUDE.md: one evaluation path).
- **One reader.** `perinv.eval.warehouse.connect()` returns a DuckDB connection with the views in §5.
  Do not glob the files yourself.
- **Scope of v1.** Only final evaluations on the `test` CRN split. Tuning trials, learning curves and
  model checkpoints stay as files in the run directory and are not part of the warehouse.

## 2. Layout
Everything for a run lives in its run directory, so there is exactly one copy of each result.
```
runs/<run_name>/
  manifest.json        # lineage + status (§4), the run dimension
  config.yaml          # resolved config, human-readable copy
  model.zip, vecnormalize.pkl, learning_curve.csv   # training outputs (M2), if any
  eval_test.parquet    # fact_step for this run (§3)
```
- `run_name` = `<group>-N<N>-k<κ>-<policy>-s<seed>`, e.g. `A-N1-k0.6-ppo-s0`. Readable, used as the folder name.
- **No hive partitioning.** Core experiments total about 0.8M rows, about 10 MB (§7), so skipping
  files by partition saves almost nothing. Folder-name partitions also caused a silent bug in the
  spike: DuckDB read `kappa=0.6` as text, and `WHERE kappa = 1` matched the 0.6 rows too. Filters
  come from joining `dim_run` instead.
- `runs/` is gitignored.

## 3. `fact_step`: one row per (run, path, day, SKU)
Grain: `(run_key, path_id, t, sku)`, which is unique. Rows are sorted by `path_id, t, sku`, and the file is compressed with zstd.

| Column | Type | Meaning (SPEC §3 step numbers) |
|---|---|---|
| `run_key` | string (dictionary) | Idempotency key of the run (§4). The same value on every row |
| `path_id` | int16 | Index k of the `test` CRN path |
| `t` | int16 | Day, 0 … T−1 |
| `sku` | int16 | SKU index, 0 … N−1 |
| `occ` | int32 | Occupancy of this SKU at decision time (step 1, before ordering): `sum(inv[i]) + sum(pipe[i])` |
| `order` | int32 | Units that actually entered the pipeline (after clipping) |
| `sold` | int32 | Step 3 |
| `lost` | int32 | Step 3. Demand = `sold + lost` |
| `expired` | int32 | Step 4 |
| `held` | int32 | Step 5, on hand after expiry |
| `day_reward` | float64, **sku = 0 row only** | The env's total reward for the day (SPEC §3 step 7). NULL on other rows |
| `clipped` | bool, **sku = 0 row only** | The env clipped today's order (`info["clipped"]`). NULL on other rows |

Why these choices:
- **Quantities only, no money columns.** Revenue and costs equal quantity × unit price, so the views
  compute them from the per-SKU prices in `dim_run`. Storing them would duplicate data and add float
  rounding. Every column except `day_reward` is a small integer, which compresses very well.
- **Day-level values on the sku = 0 row only.** If `day_reward` were repeated on every SKU row,
  `SUM(day_reward)` would silently count it N times. With NULLs on the other rows, `SUM` and `COUNT`
  are correct without any special handling.
- **`day_reward` is kept for reconciliation.** The per-SKU reward recomputed from quantities must
  match the env's own reward. If it does not, the env, the writer or the cost parameters have a bug (§6).

Parquet key-value metadata on every file: `perinv.eval_schema_version`, `perinv.run_key`, and
`perinv.n_paths`/`T`/`N`. Files describe themselves even when they are copied out of their folder.

## 4. Run identity, lineage and status (`manifest.json` = `dim_run`)

### 4.1 Keys: identity is separate from lineage
| Key | Built from | Used for |
|---|---|---|
| `train_key` | env `config_hash`, policy name + hash of resolved hyperparameters (tuned values included), seed, `train_env_steps`, `EVENT_ORDER_VERSION` | Reusing a trained model. If it matches, skip training |
| `run_key` | `train_key` + the eval world's `config_hash`, CRN `demand_hash`, sha256 of `test.npz` + `EVAL_SCHEMA_VERSION` | Deciding whether an evaluation is done. It is the key on every fact row |

Both are `stable_hash(...)` from `perinv.env.config` (canonical JSON → sha256, 12 hex characters).

**Not in the keys, only recorded in the manifest:** git SHA, dirty flag, host, timestamps, library
versions. A key holds only what changes the *result*. If git SHA were in the key, every commit
(even a docs typo) would give every run a new key, "skip if done" would never fire, and the whole
matrix would rerun. Changing a pinned library that can change results (torch) is a deliberate team
decision; rerun with `--force` then.

Because the keys are separate, a new eval schema re-evaluates saved models in minutes without retraining.

### 4.1b Cross-evaluation (build on config A, score on config B)
A matrix group may pair `envs` with `eval_envs`. The policy is built on A (its belief) and evaluated on B's `test` paths (the real world). A and B may differ only in `demand_mean` and `season`; anything else is rejected when the matrix is expanded. `train_key` comes from A, so a cross-evaluation reuses the model of the native run on A instead of training again (`ctx.model_dir`). The keys and the eval file describe B. The manifest adds `eval_env_path`, `eval_config_hash`, `eval_demand_hash`, `eval_demand_mean` and `cross_eval`.

### 4.2 Required manifest fields
`schema_version`, `run_name`, `run_key`, `train_key`, `status`, `group`, `policy`, `seed`, `n_skus`,
`kappa`, `capacity`, `horizon`, `env` (the full EnvConfig, with per-SKU lists `price`, `order_cost`,
`holding_cost`, `expiry_cost`, `lost_sales_cost`), `config_hash`, `demand_hash`, `crn_test_sha256`,
`train_env_steps`, `actual_env_steps`, `git_sha`, `git_dirty`, `python`, `pip_freeze`, `host`,
`start_time`, `end_time`, `cpu_seconds`, `wall_seconds`, `artifacts` (relative paths).
On failure, also `error` = {`stage`, `type`, `message`} and a `failure.json` with the same content.

### 4.3 Status lifecycle and "done"
`running` → `done` | `failed`. A run counts as **done** only if all three hold:
1. `manifest.status == "done"`
2. `manifest.run_key` equals the key computed from the current config
3. `eval_test.parquet` exists, and its metadata `run_key` matches

Write order makes crashes safe:
1. Write `eval_test.parquet.tmp`, then `os.replace` it to the final name.
2. Only after that, set the manifest status to `done`, written the same way (tmp + replace).

A crash at any point leaves either no final file, or a file whose manifest still says `running`.
Both count as not done, and the next run overwrites them. If the key no longer matches (the config
changed), the run is stale, and the runner reruns and replaces it.

## 5. Views (`perinv.eval.warehouse`, SQL in `sql/`)
Only runs that are **done** appear in the views. The reader checks that the manifest says `done`, that the eval file exists, and that the file's `run_key` and schema version agree with the manifest. Whether a run is still current for today's config is the runner's question (§4.3), not the reader's. There is also `v_day` (run × path × day: total `occ`, `occ + order`), which the utilisation metric uses.

| View | Grain | Adds |
|---|---|---|
| `dim_run` | run | The manifest fields above |
| `fact_step` | run × path × t × sku | The raw rows |
| `v_step` | run × path × t × sku | `demand`, plus revenue/ordering/holding/expiry/lost_sales and `sku_reward`, computed with that SKU's prices (`list_extract(price, sku + 1)`) |
| `v_path` | run × path | `profit = SUM(day_reward)`, plus total ordered/sold/lost/expired/demand |
| `v_run` | run | Metrics in §5.1, with `group, n_skus, kappa, policy, seed` |

### 5.1 Metric definitions (pooled over all paths, days and SKUs of a run)
- **Mean profit** = mean over paths of `v_path.profit`.
- **Waste rate** = `SUM(expired) / SUM(order)`.
- **Fill rate** = `SUM(sold) / SUM(sold + lost)`.
- **Capacity utilisation** = mean over (path, t) of `SUM_sku(occ) / capacity`. This is decision-time occupancy. Post-order occupancy is `occ + order`.
- **% of DP** (N=1) = `mean(profit_policy) / mean(profit_dp)` over the same `path_id`s and the same eval world (`eval_config_hash`). The reference is the DP evaluated natively in that world.
  It is NULL when the DP's mean profit is ≤ 0, because the ratio then flips sign or blows up. Use `mean_gap_to_dp` instead.
- **Ratio of sums, not mean of per-path ratios.** A path with a tiny denominator (e.g. 2 units
  ordered) would swing a mean of ratios a lot. Pooling weighs each unit equally.
- IQM and bootstrap CIs over seeds are computed in Python (`perinv.eval.stats`) on top of `v_run`.

## 6. Checks
The writer checks these before it writes anything, and raises instead of writing a bad file:
1. Row count = `n_paths × T × N`, and the grain is unique.
2. All quantities ≥ 0, and `sold + lost` equals the CRN demand of that path, day and SKU.
3. Every day: `SUM_sku(occ + order) ≤ capacity`.
4. Every day: `|SUM_sku(sku_reward) − day_reward| ≤ 1e-6`, where sku_reward is recomputed from quantities and the cost parameters.

The reader checks:
- Every file's `perinv.eval_schema_version` equals the current version. It refuses mixed versions.
- Tests: a query by `kappa` / `n_skus` returns the expected row count (the regression guard for the
  spike bug), and shuffling the path order gives identical per-path results.

## 7. Size (estimate, update after the first real runs)
Core rows ≈ Σ runs × 100 paths × 30 days × N:
- Group A: 50 RL runs × 3,000 rows
- Group B: 30 runs × 6,000 rows (N=2) and 30 runs × 15,000 rows (N=5)
- Plus DP and baselines

That is about 0.8M rows. The spike measured 12.2 bytes/row on random synthetic data (a pessimistic
case, because random values barely compress), so about 10 MB total. N=10 and N=20 would add a few
million rows, still well under 100 MB.

## 8. Needs from other members
- **M1 (env).**
  - `reset()` must also return `info["state"]` (`inv`, `pipe`, `t`); evaluate reads decision-time occupancy from it.
  - `info` must expose per-SKU arrays `sold, lost, expired, held, ordered` and decision-time per-SKU occupancy.
  - Add an `EVENT_ORDER_VERSION` constant in `perinv.env`.
  - Confirm whether `info["occupancy"]` is measured before or after the order.
- **M2 (train) and baseline owners: the builder contract.** The runner (`scripts/run_matrix.py`) calls
  `build(ctx) -> Policy` at the locations in `perinv.runner.DEFAULT_BUILDERS`:
  - `perinv.train.train:build` for the RL algorithms
  - `perinv.baselines.{dp_policy,s_S,newsvendor}:build` for DP and the heuristics

  `ctx` carries `cfg, policy, hparams, seed, train_env_steps, run, make_env, reuse_model, model_dir`. Build from `ctx.cfg` only: never look at the eval world. An RL builder does this:
  1. If `ctx.reuse_model` is true, load `model.zip` / `vecnormalize.pkl` from `ctx.model_dir`, which may be another run's folder.
  2. Otherwise, train and save those files into `ctx.run.dir`.
  3. Call `ctx.run.update(train={"train_key": ctx.run.train_key, "status": "done"}, actual_env_steps=...)`.
  4. Return a Policy with frozen VecNormalize.

  The runner takes care of seeding, the manifest, evaluation and writing the result.
