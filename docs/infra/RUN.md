# M3 infra: how to run

Commands are for Git Bash on Windows, run from the repo root. Data lives in `FRN_DATA_DIR` (default `D:/NEU/data`). On Linux/macOS, drop `MSYS_NO_PATHCONV=1` and use `.venv/bin/python`.

## 1. Setup (once)

```bash
# Team env (Python 3.15). PYYAML_FORCE_LIBYAML=0 is needed only when no C compiler is installed (Windows).
uv venv .venv --python 3.15.2
PYYAML_FORCE_LIBYAML=0 uv pip install --python .venv -r requirements.txt -r requirements-dev.txt -e .

# Data platform image (Docker Desktop must be running)
docker compose -f infra/compose.yaml build
```

## 2. Experiments (team side)

```bash
.venv/Scripts/python scripts/make_crn_paths.py configs/env/*.yaml                 # CRN demand paths
.venv/Scripts/python scripts/run_matrix.py configs/experiments/core.yaml --dry-run  # what would run
.venv/Scripts/python scripts/run_matrix.py configs/experiments/core.yaml --jobs 6   # run (safe to rerun)
.venv/Scripts/python scripts/analyze.py                                             # results/ tables + plots
```

Useful flags: `--seeds 0,1,2`, `--groups A`, `--policies ppo,dp`, `--profile smoke`, `--force`, `--official`.
If a run failed, read `runs/<run_name>/failure.json`.

## 3. Real data → demand parameters (infra)

```bash
cd infra
MSYS_NO_PATHCONV=1 docker compose run --rm frn dagster asset materialize -m frn.definitions --select '*'
```

Each layer runs only if its inputs changed. On a fresh machine the first run takes about 3 minutes (115 MB download, then bronze about 2 min).

Run a single layer: `docker compose run --rm frn python -m frn.<ingest|bronze|silver|gold> --data-root /data/freshretailnet` (prefix with `MSYS_NO_PATHCONV=1`).
Gold options: `--censor-threshold 2` (stockout hours that make a day censored), `--workers 4`.

UI (lineage, checks, run history):

```bash
docker compose -f infra/compose.yaml up -d dagster     # http://localhost:3000
docker compose -f infra/compose.yaml stop dagster
```

## 4. Calibrated env configs (back on the team side)

```bash
.venv/Scripts/python scripts/make_calibrated_configs.py D:/NEU/data/freshretailnet/gold/c374e85975c82816c495acfa904cc55ab14a47d3/demand_params.parquet
.venv/Scripts/python scripts/make_crn_paths.py configs/env/calibrated/*.yaml
.venv/Scripts/python scripts/run_matrix.py configs/experiments/calibrated.yaml --dry-run
```

Question 4 (once the builders exist): run the group, then compare in Python.

```python
from perinv.eval import metrics, warehouse
con = warehouse.connect("runs")
metrics.paired_groups(con, "D13", "D3")   # profit lost by calibrating with method 1; negative = cost
```

## 5. Tests

```bash
.venv/Scripts/python -m pytest -q             # team side (slow kill/resume test: -m slow)
docker compose -f infra/compose.yaml run --rm test   # infra side
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| A container path becomes `C:/Program Files/Git/...` | Prefix the docker command with `MSYS_NO_PATHCONV=1` |
| PyYAML build asks for Visual C++ | `PYYAML_FORCE_LIBYAML=0` before `pip install` |
| `no CRN paths for ...` | Run `make_crn_paths.py` for those env configs |
| Every job shows `blocked (no builder yet)` | Expected until `perinv.train.train:build` and the baseline builders exist |
| A run is `failed` | Read `runs/<run_name>/failure.json` (`stage`, `type`, `message`); `WorkerCrashed` = the process died (out of memory?) |
| `eval schema vX, reader expects vY` | Rerun the matrix: stale runs are re-evaluated automatically |
