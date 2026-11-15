"""Bronze layer: raw Parquet -> typed Parquet. Types only, meaning unchanged; every known fact is a check.

    python -m frn.bronze --data-root /data/freshretailnet

In:  <data-root>/raw/<revision>/data/{train,eval}.parquet  (+ raw _manifest.json from frn.ingest)
Out: <data-root>/bronze/<revision>/{train,eval}.parquet + _manifest.json

- Schema contract: the raw columns and types must match RAW_SCHEMA exactly, or nothing runs (upstream drift).
- Checks: run them all, record each result in the manifest, and write nothing if any of them fails.
- Idempotent: skipped when the bronze manifest already records the same raw hashes and BRONZE_VERSION.
- Atomic: each file goes to `.tmp` first and is renamed only after every check has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from frn.ingest import Source

log = logging.getLogger("frn.bronze")

BRONZE_VERSION = 3   # bump when the transform or the checks change; forces a rebuild
SPLITS = ("train", "eval")
HOURS = 24
# Hours counted by stock_hour6_22_cnt: 0-based hours 6..21 (16 hours). DuckDB lists are 1-based: [7:22].
# Verified on the eval split on 2026-11-14 (100% of rows; [7:23] and [6:22] matched only 62% / 84%).
OPEN_SLICE = "[7:22]"

RAW_SCHEMA = {
    "city_id": "BIGINT", "store_id": "BIGINT", "management_group_id": "BIGINT", "first_category_id": "BIGINT",
    "second_category_id": "BIGINT", "third_category_id": "BIGINT", "product_id": "BIGINT", "dt": "VARCHAR",
    "sale_amount": "DOUBLE", "hours_sale": "DOUBLE[]", "stock_hour6_22_cnt": "INTEGER",
    "hours_stock_status": "BIGINT[]", "discount": "DOUBLE", "holiday_flag": "INTEGER", "activity_flag": "INTEGER",
    "precpt": "DOUBLE", "avg_temperature": "DOUBLE", "avg_humidity": "DOUBLE", "avg_wind_level": "DOUBLE",
}
_IDS = ("city_id", "store_id", "management_group_id", "first_category_id", "second_category_id",
        "third_category_id", "product_id")

# The bronze projection: same rows, same meaning, tighter and explicit types. Sorted for locality + compression.
TRANSFORM = f"""
SELECT
    {", ".join(f"{c}::INTEGER AS {c}" for c in _IDS)},
    strptime(dt, '%Y-%m-%d')::DATE              AS dt,
    sale_amount,
    hours_sale,
    stock_hour6_22_cnt::SMALLINT                AS stock_hour6_22_cnt,
    hours_stock_status::TINYINT[]               AS hours_stock_status,  -- 1 = out of stock (checked)
    discount,
    holiday_flag = 1                            AS holiday_flag,
    activity_flag = 1                           AS activity_flag,
    precpt, avg_temperature, avg_humidity, avg_wind_level
FROM src
ORDER BY store_id, product_id, dt
"""

# (name, SQL returning one value, predicate on that value). Every check runs on the raw data of one split.
# (name, aggregate expression over src, predicate on its value[, severity="error"]).
# All expressions go into ONE select, so the raw file is scanned once (15 separate scans took 132 s on train).
CHECKS: list[tuple] = [
    ("rows > 0", "count(*)", lambda v: v > 0),
    ("complete panel: rows == series x dates",
     "count(*) - count(DISTINCT (store_id, product_id)) * count(DISTINCT dt)", lambda v: v == 0),
    ("unique key (store_id, product_id, dt)", "count(*) - count(DISTINCT (store_id, product_id, dt))",
     lambda v: v == 0),
    ("no nulls", " + ".join(f'(count(*) - count("{c}"))' for c in RAW_SCHEMA), lambda v: v == 0),
    ("dt parses as YYYY-MM-DD", "count(*) FILTER (WHERE try_strptime(dt, '%Y-%m-%d') IS NULL)", lambda v: v == 0),
    ("ids fit in INTEGER", "max(greatest(" + ", ".join(f"abs({c})" for c in _IDS) + "))", lambda v: v < 2**31),
    (f"{HOURS} hourly values per row",
     f"count(*) FILTER (WHERE len(hours_sale) <> {HOURS} OR len(hours_stock_status) <> {HOURS})", lambda v: v == 0),
    ("stock status is 0/1",
     "count(*) FILTER (WHERE list_min(hours_stock_status) < 0 OR list_max(hours_stock_status) > 1)",
     lambda v: v == 0),
    ("flags are 0/1", "count(*) FILTER (WHERE holiday_flag NOT IN (0, 1) OR activity_flag NOT IN (0, 1))",
     lambda v: v == 0),
    ("sales >= 0", "count(*) FILTER (WHERE sale_amount < 0 OR list_min(hours_sale) < 0)", lambda v: v == 0),
    ("sale_amount == sum(hours_sale)", "max(abs(sale_amount - list_sum(hours_sale)))", lambda v: v <= 1e-9),
    ("stock_hour6_22_cnt == stockout hours 6..21",
     f"count(*) FILTER (WHERE stock_hour6_22_cnt <> list_sum(hours_stock_status{OPEN_SLICE}))", lambda v: v == 0),
    # Semantic guard: if 1 meant "in stock", hours flagged 1 would sell *more* often, not far less.
    # A scalar subquery on a fixed-seed sample, evaluated inside the same statement.
    ("status 1 means out of stock (sales/hour much rarer when 1)",
     """(SELECT avg((s > 0)::INT) FILTER (WHERE st = 1) / avg((s > 0)::INT) FILTER (WHERE st = 0)
         FROM (SELECT unnest(hours_sale) s, unnest(hours_stock_status) st
               FROM src USING SAMPLE reservoir(20000 ROWS) REPEATABLE (42)))""",
     lambda v: v < 0.5),
    # Discount: 1.0 = full price. Values slightly above 1 exist in the real data (one product, 34 train rows,
    # up to 1.088: price briefly above list), and 0.0 appears ~16k times (likely "unknown", not "free").
    # Neither changes demand estimation, so they are warnings; only absurd values stop the build.
    ("discount in [0, 2]", "count(*) FILTER (WHERE discount < 0 OR discount > 2)", lambda v: v == 0),
    ("discount in [0, 1]", "count(*) FILTER (WHERE discount < 0 OR discount > 1)", lambda v: v == 0, "warn"),
    ("discount == 0 rows (likely missing, not free)", "count(*) FILTER (WHERE discount = 0)", lambda v: v == 0,
     "warn"),
]


class BronzeCheckError(RuntimeError):
    """Raw data broke the schema contract or a check; nothing was written."""


@dataclass
class CheckResult:
    name: str
    value: float | int | None
    passed: bool
    severity: str = "error"  # "error" stops the build; "warn" is recorded and logged only


def _schema_diff(con: duckdb.DuckDBPyConnection) -> list[str]:
    got = {name: typ for name, typ, *_ in con.execute("DESCRIBE src").fetchall()}
    problems = [f"missing column {c} ({t})" for c, t in RAW_SCHEMA.items() if c not in got]
    problems += [f"unexpected column {c} ({t})" for c, t in got.items() if c not in RAW_SCHEMA]
    problems += [f"{c}: type {got[c]}, expected {t}" for c, t in RAW_SCHEMA.items() if c in got and got[c] != t]
    return problems


def run_checks(con: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    select = ", ".join(expr for _, expr, *_ in CHECKS)
    values = con.execute(f"SELECT {select} FROM src").fetchall()[0]
    out = []
    for (name, _, predicate, *rest), v in zip(CHECKS, values, strict=True):
        passed = v is not None and bool(predicate(v))
        out.append(CheckResult(name, v if not isinstance(v, float) else round(v, 12), passed, *rest))
    return out


def _input_key(raw_files: dict[str, str]) -> str:
    blob = json.dumps({"raw_sha256": raw_files, "bronze_version": BRONZE_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build(data_root: str | Path, source: Source | None = None, memory_limit: str = "4GB") -> Path:
    """Build bronze for every split. Returns the bronze directory. Raises BronzeCheckError on bad input."""
    source = source or Source.load()
    data_root = Path(data_root)
    raw_dir = data_root / "raw" / source.revision
    raw_manifest = json.loads((raw_dir / "_manifest.json").read_text(encoding="utf-8"))
    if raw_manifest["revision"] != source.revision:
        raise BronzeCheckError(f"raw manifest is for {raw_manifest['revision']}, source pins {source.revision}")
    raw_sha = {s: raw_manifest["files"][f"data/{s}.parquet"]["sha256"] for s in SPLITS}
    key = _input_key(raw_sha)

    out_dir = data_root / "bronze" / source.revision
    manifest_path = out_dir / "_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("input_key") == key and all((out_dir / f"{s}.parquet").exists() for s in SPLITS):
            log.info("bronze up to date (input_key %s), skipping", key)
            return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{(out_dir / '.duckdb_tmp').as_posix()}'")
    results, stats, failures = {}, {}, []
    for split in SPLITS:
        src = (raw_dir / "data" / f"{split}.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet('{src}')")
        drift = _schema_diff(con)
        if drift:
            raise BronzeCheckError(f"{split}: raw schema drift: " + "; ".join(drift))
        t0 = time.perf_counter()
        res = run_checks(con)
        results[split] = [r.__dict__ for r in res]
        failures += [f"{split}: {r.name} (got {r.value})" for r in res if not r.passed and r.severity == "error"]
        for r in res:
            if not r.passed and r.severity == "warn":
                log.warning("%s: %s (got %s)", split, r.name, r.value)
        stats[split] = {"check_seconds": round(time.perf_counter() - t0, 2)}
    # Cross-split check: eval must be strictly after train (no leakage between splits).
    gap = con.execute(f"""SELECT (SELECT min(dt) FROM read_parquet('{(raw_dir / 'data/eval.parquet').as_posix()}'))
                              > (SELECT max(dt) FROM read_parquet('{(raw_dir / 'data/train.parquet').as_posix()}'))
                       """).fetchall()[0][0]
    if not gap:
        failures.append("eval dates are not strictly after train dates")
    if failures:
        raise BronzeCheckError("bronze checks failed, nothing written:\n  " + "\n  ".join(failures))

    for split in SPLITS:
        src = (raw_dir / "data" / f"{split}.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet('{src}')")
        final, tmp = out_dir / f"{split}.parquet", out_dir / f"{split}.parquet.tmp"
        t0 = time.perf_counter()
        try:
            con.execute(f"COPY ({TRANSFORM}) TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
            os.replace(tmp, final)
        finally:
            tmp.unlink(missing_ok=True)
        rows, d0, d1, series = con.execute(f"SELECT count(*), min(dt), max(dt), count(DISTINCT (store_id, product_id)) "
                                           f"FROM read_parquet('{final.as_posix()}')").fetchall()[0]
        stats[split] |= {"rows": rows, "series": series, "date_min": str(d0), "date_max": str(d1),
                         "bytes": final.stat().st_size, "write_seconds": round(time.perf_counter() - t0, 2)}
        log.info("bronze %s: %s rows, %.1f MB", split, f"{rows:,}", final.stat().st_size / 1e6)

    manifest = {"layer": "bronze", "bronze_version": BRONZE_VERSION, "input_key": key, "revision": source.revision,
                "raw_sha256": raw_sha, "built_at": datetime.now(UTC).isoformat(), "duckdb": duckdb.__version__,
                "stats": stats, "checks": results}
    tmp = manifest_path.with_name("_manifest.json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the bronze layer from the verified raw files.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--memory-limit", default="4GB")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        out = build(args.data_root, memory_limit=args.memory_limit)
    except BronzeCheckError as exc:
        log.error("%s", exc)
        return 2
    print(f"bronze ready at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
