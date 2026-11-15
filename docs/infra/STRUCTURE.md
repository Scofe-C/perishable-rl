# M3 infra: structure

Two parts. They share no code; they share **files**.

| Part | Where | Python | Job |
|---|---|---|---|
| Experiment runtime | `src/perinv/` (team repo) | 3.15, team env + `duckdb` | Run the experiment matrix, store every evaluation, compute metrics |
| Data platform | `infra/` (Docker) | 3.12, own lock file | FreshRetailNet → per-SKU demand parameters (real data, stockout-aware) |

## Data flow

```
Hugging Face FreshRetailNet-50K (pinned commit)
        │
        ▼   infra/  (Docker; orchestrated by Dagster)
   raw ──► bronze ──► silver (dbt) ──► gold
                                        │
                                        ▼
                       demand_params.parquet   (contract C1)
                                        │
        ┌───────────────────────────────┘
        ▼   team repo
 scripts/make_calibrated_configs.py ──► configs/env/calibrated/*.yaml
                                        │
 configs/env/*.yaml ─────────────────────┤
                                        ▼
 scripts/run_matrix.py ──► runs/<run_name>/{manifest.json, eval_test.parquet}
                                        │
                                        ▼
 perinv.eval.warehouse.connect() ──► SQL views ──► scripts/analyze.py ──► results/
```

## Experiment runtime (`src/perinv/`, team repo)

| File | Does |
|---|---|
| `runner.py` + `scripts/run_matrix.py` | Matrix YAML → jobs; skips finished runs; K parallel workers; resumes after a kill; cross-evaluation (`eval_envs`: build on A, score on B, reuse A's model) |
| `utils/run_io.py` | `train_key` / `run_key`; manifest; status `running → done/failed`; `failure.json`; `log.jsonl` |
| `utils/atomic.py` | Write to `.tmp`, then rename (no half-written files) |
| `eval/evaluate.py` | The only evaluation loop; one seed per CRN path |
| `eval/writer.py` | Checks, then writes `eval_test.parquet` (one row per run × path × day × SKU) |
| `eval/warehouse.py` + `eval/sql/views.sql` | DuckDB over finished runs: `dim_run`, `fact_step`, `v_step`, `v_path`, `v_day`, `v_run`, `v_vs_dp` |
| `eval/metrics.py`, `eval/stats.py` | Report tables; IQM + bootstrap CI; paired per-path differences |
| `scripts/analyze.py` | `runs.csv`, `summary.csv`, `profit_<group>.png` |
| `data/calibration.py` + `scripts/make_calibrated_configs.py` | C1 file → env YAML (SKU selection, scale, capacity) |
| `configs/experiments/core.yaml`, `calibrated.yaml` | Matrices: core (124 jobs); group D (24 jobs: D1, D3, D13 cross-evaluation, DP at N=1) |

Contract: [WAREHOUSE.md](../WAREHOUSE.md). Members plug in by writing `build(ctx) -> Policy` (WAREHOUSE.md §8).

## Data platform (`infra/`)

| Layer | Code | Output (`D:\NEU\data\freshretailnet\<layer>\<revision>\`) | Guarantee |
|---|---|---|---|
| raw | `src/frn/ingest.py` | original files + `_manifest.json` | URL pinned to a commit; sha256 must match `configs/source.yaml` |
| bronze | `src/frn/bronze.py` | `train.parquet`, `eval.parquet` | Types only; schema contract; 16 checks (error/warn); nothing written on error |
| silver | `dbt/` + `src/frn/silver.py` | `daily`, `hour_profile`, `demand_m2` (Parquet) | dbt builds in staging; 24 tests; published only if all pass |
| gold | `src/frn/gold.py`, `censored_nb.py` | `demand_params.parquet` | Same NB model for methods 1/2/3; C1 checks; held-out scoring |
| orchestration | `src/frn/definitions.py` | run history in `D:\NEU\data\dagster_home` | 4 assets + 4 checks; UI at localhost:3000 |

Every layer: skip when inputs are unchanged, atomic publish, manifest with input hashes.

## File contracts

| File | Producer → consumer | Key rule |
|---|---|---|
| `demand_params.parquet` (C1) | infra gold → `make_calibrated_configs.py` | one row per (sku_id, method); `lambda_base > 0`; dow factors mean 1 |
| `configs/env/*.yaml` | calibration / hand-written → runner, env | loads with `perinv.env.config.load_env_config` |
| `data/crn/<hash>/{tune,test}.npz` | `make_crn_paths.py` → evaluate | sha256 in manifest; never tune on `test` |
| `runs/<run>/manifest.json` | runner → warehouse | `status`, `run_key`, `train_key`, git SHA, env costs |
| `runs/<run>/eval_test.parquet` | evaluate → warehouse | schema version in file metadata; integer quantities only |

## Dependencies

- Team env: everything in `requirements.txt`; this work added only `duckdb==2.0.0`.
- infra env: `infra/pyproject.toml` + `infra/uv.lock` (dagster 1.13.29, dbt-core 1.11.17, dbt-duckdb 1.11.1, duckdb 1.5.6, scipy, pyarrow).
- No imports across the two. Telemetry off for dbt and Dagster.

## Known limits

- Runs need `build()` functions from M1/M2 (env, training, baselines). Until then every job shows `blocked`.
- Capacity uses the provisional rule (SPEC §1.1). Replace it with O_ref once that is measured.
- Real sales are normalized (coefficient unknown). Only ratios are used, with one explicit scale.
- Gold results match across Windows and Linux to about 0.02%, not bit for bit.
