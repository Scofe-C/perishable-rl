"""Silver layer: run the dbt project (bronze -> silver) with write-audit-publish.

    python -m frn.silver --data-root /data/freshretailnet

1. Write:   `dbt build` materializes every silver model into a fresh staging folder.
2. Audit:   the same `dbt build` runs every data test; any failure aborts. Nothing is published.
3. Publish: the staging folder replaces <data-root>/silver/<revision>/ by renaming folders.
Readers see the old complete version or the new complete version, never a mix of the two.

Skipped when the silver manifest already records the same bronze input, dbt project files and vars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from frn.ingest import Source
from frn.publish import publish_dir

log = logging.getLogger("frn.silver")
DBT_DIR = Path(__file__).resolve().parents[2] / "dbt"
OUTPUTS = ("daily.parquet", "hour_profile.parquet", "demand_m2.parquet")
KEEP_DBT_RUNS = 5


class SilverBuildError(RuntimeError):
    """dbt failed (a model error or a failing data test); nothing was published."""


def project_fingerprint(dbt_dir: Path = DBT_DIR) -> str:
    """Hash of every file that defines the transformation (models, tests, macros, project config)."""
    h = hashlib.sha256()
    for p in sorted(dbt_dir.rglob("*")):
        if p.is_file() and p.suffix in {".sql", ".yml", ".yaml"} and "target" not in p.parts and "logs" not in p.parts:
            h.update(p.relative_to(dbt_dir).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))  # same hash on Windows and Linux checkouts
    return h.hexdigest()[:16]


def _run_dbt(args: list[str]) -> list[dict]:
    """Invoke dbt in-process. Returns per-node results; raises SilverBuildError if anything failed."""
    from dbt.cli.main import dbtRunner

    res = dbtRunner().invoke(args)
    rows = []
    if res.result is not None and hasattr(res.result, "results"):
        rows = [{"node": r.node.unique_id, "status": str(r.status), "failures": getattr(r, "failures", None),
                 "seconds": round(r.execution_time, 2), "message": r.message} for r in res.result.results]
    if not res.success:
        bad = [r for r in rows if r["status"] not in ("success", "pass")]
        detail = "; ".join(f"{r['node']}: {r['status']} {r['message'] or ''}".strip() for r in bad[:10])
        raise SilverBuildError(f"dbt build failed: {detail or res.exception}")
    return rows


def build(data_root: str | Path, source: Source | None = None, dbt_vars: dict | None = None) -> Path:
    source = source or Source.load()
    data_root = Path(data_root)
    bronze_manifest = json.loads((data_root / "bronze" / source.revision / "_manifest.json").read_text("utf-8"))
    extra_vars = dbt_vars or {}
    input_key = hashlib.sha256(json.dumps({"bronze": bronze_manifest["input_key"], "dbt": project_fingerprint(),
                                           "vars": extra_vars}, sort_keys=True).encode()).hexdigest()[:16]

    final = data_root / "silver" / source.revision
    manifest_path = final / "_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text("utf-8")).get("input_key") == input_key:
        log.info("silver up to date (input_key %s), skipping", input_key)
        return final

    build_id = uuid.uuid4().hex[:8]
    staging = data_root / "silver" / f"_staging_{source.revision[:12]}_{build_id}"
    staging.mkdir(parents=True)
    # dbt logs and compiled SQL live OUTSIDE staging: in-process dbt keeps its log file open, and Windows
    # refuses to rename a folder that holds an open file. Only run_results.json is copied into the product.
    dbt_work = data_root / "silver" / "_dbt_runs" / build_id
    dbt_vars_all = {"data_root": data_root.as_posix(), "revision": source.revision, "out_dir": staging.as_posix(),
                    **extra_vars}
    os.environ["FRN_DBT_DB"] = (staging / "_build.duckdb").as_posix()
    os.environ["DBT_SEND_ANONYMOUS_USAGE_STATS"] = "false"
    t0 = time.perf_counter()
    try:
        results = _run_dbt(["build", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR),
                            "--target-path", str(dbt_work / "target"), "--log-path", str(dbt_work / "logs"),
                            "--vars", json.dumps(dbt_vars_all), "--no-use-colors"])
        missing = [f for f in OUTPUTS if not (staging / f).exists()]
        if missing:
            raise SilverBuildError(f"dbt finished but outputs are missing: {missing}")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    (staging / "_build.duckdb").unlink(missing_ok=True)  # scratch database, not a product
    (staging / "_build.duckdb.wal").unlink(missing_ok=True)
    shutil.copy2(dbt_work / "target" / "run_results.json", staging / "_dbt_run_results.json")

    manifest = {
        "layer": "silver", "input_key": input_key, "revision": source.revision,
        "bronze_input_key": bronze_manifest["input_key"], "dbt_project_fingerprint": project_fingerprint(),
        "vars": dbt_vars_all | {"out_dir": "<staging>"}, "built_at": datetime.now(UTC).isoformat(),
        "seconds": round(time.perf_counter() - t0, 1),
        "outputs": {f: {"bytes": (staging / f).stat().st_size} for f in OUTPUTS},
        "tests": {"total": sum(r["node"].startswith("test.") for r in results),
                  "passed": sum(r["node"].startswith("test.") and r["status"] == "pass" for r in results)},
        "nodes": results,
    }
    (staging / "_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    publish_dir(staging, final)
    runs = sorted(dbt_work.parent.iterdir(), key=lambda p: p.stat().st_mtime)
    for stale in runs[:-KEEP_DBT_RUNS]:  # dbt logs are for debugging; keep only the most recent builds
        shutil.rmtree(stale, ignore_errors=True)
    log.info("silver published at %s (%d/%d tests passed)", final, manifest["tests"]["passed"],
             manifest["tests"]["total"])
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + test + publish the silver layer (dbt).")
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        out = build(args.data_root)
    except SilverBuildError as exc:
        log.error("%s", exc)
        return 2
    print(f"silver ready at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
