"""The one reader of the evaluation warehouse. Contract: docs/WAREHOUSE.md §5 and §6.

    con = connect("runs")
    con.sql("SELECT policy, AVG(mean_profit) FROM v_run GROUP BY policy").df()

Only runs that are done are visible: manifest status "done", eval file present, and the file's run_key and
schema version agree with the manifest. Column types are fixed here, never guessed from folder names or JSON.

Owner: Member 3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from perinv.eval.writer import EVAL_SCHEMA_VERSION, SCHEMA
from perinv.utils.run_io import EVAL_FILE

log = logging.getLogger("perinv.warehouse")
SQL_DIR = Path(__file__).parent / "sql"

_COSTS = ("price", "order_cost", "holding_cost", "expiry_cost", "lost_sales_cost", "demand_mean")
DIM_RUN_SCHEMA = pa.schema([
    ("run_key", pa.string()), ("run_name", pa.string()), ("group", pa.string()), ("policy", pa.string()),
    ("seed", pa.int32()), ("n_skus", pa.int32()), ("kappa", pa.float64()), ("capacity", pa.int32()),
    ("horizon", pa.int32()), ("config_hash", pa.string()), ("demand_hash", pa.string()),
    # the world the run was scored in (= config_hash / demand_hash unless it is a cross-evaluation)
    ("eval_config_hash", pa.string()), ("eval_demand_hash", pa.string()), ("cross_eval", pa.bool_()),
    ("crn_test_sha256", pa.string()), ("train_key", pa.string()),
    ("train_env_steps", pa.int64()), ("actual_env_steps", pa.int64()),
    ("git_sha", pa.string()), ("git_dirty", pa.bool_()), ("official", pa.bool_()),
    ("cpu_seconds", pa.float64()), ("wall_seconds", pa.float64()),
    ("start_time", pa.string()), ("end_time", pa.string()),
    *[(c, pa.list_(pa.float64())) for c in _COSTS],
])


class WarehouseError(RuntimeError):
    """The warehouse is inconsistent (e.g. mixed schema versions); fix or re-evaluate before analysing."""


def _file_meta(path: Path) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in (pq.read_schema(path).metadata or {}).items()}


def done_runs(runs_root: str | Path) -> list[tuple[dict, Path]]:
    """(manifest, eval file) for every finished run. Raises on a schema-version mismatch instead of mixing."""
    out = []
    for manifest_path in sorted(Path(runs_root).glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        f = manifest_path.parent / EVAL_FILE
        if m.get("status") != "done" or not f.exists():
            continue
        meta = _file_meta(f)
        if meta.get("perinv.run_key") != m.get("run_key"):
            log.warning("skipping %s: eval file belongs to another run_key", manifest_path.parent.name)
            continue
        if meta.get("perinv.eval_schema_version") != str(EVAL_SCHEMA_VERSION):
            raise WarehouseError(f"{f} has eval schema v{meta.get('perinv.eval_schema_version')}, reader expects "
                                 f"v{EVAL_SCHEMA_VERSION}; re-evaluate it (the runner reruns stale runs)")
        out.append((m, f))
    return out


def _dim_row(m: dict) -> dict:
    env = m.get("env") or {}
    row = {name: m.get(name) for name in DIM_RUN_SCHEMA.names if name not in _COSTS}
    row["eval_config_hash"] = m.get("eval_config_hash") or m.get("config_hash")   # manifests before cross-eval
    row["eval_demand_hash"] = m.get("eval_demand_hash") or m.get("demand_hash")
    row["cross_eval"] = bool(m.get("cross_eval", False))
    row.update({c: [float(x) for x in env.get(c, ())] for c in _COSTS})
    return row


def connect(runs_root: str | Path = "runs", con: duckdb.DuckDBPyConnection | None = None
            ) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with tables `dim_run`, `fact_step` and the views from sql/views.sql."""
    con = con or duckdb.connect()
    runs = done_runs(runs_root)
    dim = pa.Table.from_pylist([_dim_row(m) for m, _ in runs], schema=DIM_RUN_SCHEMA)
    con.register("_dim_run", dim)
    con.execute("CREATE OR REPLACE TABLE dim_run AS SELECT * FROM _dim_run")
    con.unregister("_dim_run")
    if runs:
        # An explicit file list (not a glob): only done runs, no type guessing from folder names.
        # Views cannot take query parameters, so the list is an escaped SQL literal.
        files = ", ".join("'" + f.as_posix().replace("'", "''") + "'" for _, f in runs)
        con.execute(f"CREATE OR REPLACE VIEW fact_step AS SELECT * FROM read_parquet([{files}])")
    else:
        con.register("_empty_fact", SCHEMA.empty_table())
        con.execute("CREATE OR REPLACE TABLE fact_step AS SELECT * FROM _empty_fact")
        con.unregister("_empty_fact")
    con.execute((SQL_DIR / "views.sql").read_text(encoding="utf-8"))
    log.info("warehouse: %d done runs from %s", len(runs), runs_root)
    return con
