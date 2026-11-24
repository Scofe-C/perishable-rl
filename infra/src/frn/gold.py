"""Gold layer: silver -> demand_params.parquet (contract C1), one row per (product, estimation method).

    python -m frn.gold --data-root /data/freshretailnet [--censor-threshold 1] [--workers 4]

Methods (same censored negative-binomial model, frn.censored_nb; only the data handling differs):
    drop_stockout_days    method 1: fit on days without a stockout; stockout days are dropped
    hourly_extrapolation  method 2: fit on silver demand_m2 (recovered demand); non-estimable days dropped
    censored_mle          method 3: all days; a stockout day contributes P(D >= sales)

A day counts as a stockout day when stockout_hours_open >= censor_threshold (default 1 hour).
Units: counts = round(unit_multiplier * sales), unit_multiplier = 10 (plan B: the dataset's normalization
coefficient is unknown; lambda_base is reported back in normalized units, and the env scale is a separate,
explicit parameter downstream). Every product is scored out of sample on the eval week with the censored
likelihood, the same yardstick for all three methods.
"""

from __future__ import annotations

import os

# Parallelism is one product per process. Each process must use a single BLAS thread: otherwise every worker
# starts one OpenBLAS thread per core, each reserving buffers, and on Windows the workers crashed with
# "OpenBLAS error: Memory allocation still failed". Must be set before numpy is imported (workers re-import us).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from frn import censored_nb as cnb
from frn.ingest import Source
from frn.publish import new_staging, publish_dir

log = logging.getLogger("frn.gold")
GOLD_VERSION = 1
SCHEMA_VERSION = 1
METHODS = ("drop_stockout_days", "hourly_extrapolation", "censored_mle")
MIN_OBS = 30                # a method needs at least this many training days for a product
MAX_NOT_CONVERGED = 0.05    # more non-converged fits than this stops the publish

SCHEMA = pa.schema([
    ("sku_id", pa.int32()), ("source", pa.string()), ("estimation_method", pa.string()),
    ("lambda_base", pa.float64()), *[(f"dow_factor_{d}", pa.float64()) for d in range(7)],
    ("dispersion_r", pa.float64()), ("n_obs", pa.int32()), ("n_censored", pa.int32()),
    ("converged", pa.bool_()), ("loglik", pa.float64()), ("heldout_loglik_per_day", pa.float64()),
    ("n_stores", pa.int32()), ("stockout_day_rate", pa.float64()), ("censor_threshold_hours", pa.int16()),
    ("unit_multiplier", pa.float64()), ("schema_version", pa.int16()),
])


class GoldCheckError(RuntimeError):
    """The estimates break the C1 contract; nothing was published."""


@dataclass(frozen=True)
class ProductData:
    product_id: int
    n_stores: int
    stockout_day_rate: float
    train: tuple[np.ndarray, ...]   # y, dow, cens, w (aggregated)
    m2: tuple[np.ndarray, ...]      # y, dow, w
    heldout: tuple[np.ndarray, ...] # y, dow, cens, w (eval split)


def load(silver_dir: Path, threshold: int, mult: float) -> list[ProductData]:
    """Aggregate silver into per-product (y, dow, censored, weight) arrays with DuckDB."""
    con = duckdb.connect()
    daily, m2 = (silver_dir / "daily.parquet").as_posix(), (silver_dir / "demand_m2.parquet").as_posix()
    agg = f"""
        SELECT product_id, split, round({mult} * sales)::INTEGER AS y, dow,
               stockout_hours_open >= {threshold} AS cens, count(*) AS w
        FROM read_parquet('{daily}') GROUP BY ALL ORDER BY ALL"""
    rows = con.execute(agg).fetchnumpy()
    m2rows = con.execute(f"""
        SELECT m.product_id, round({mult} * m.demand_m2)::INTEGER AS y, d.dow, count(*) AS w
        FROM read_parquet('{m2}') m JOIN read_parquet('{daily}') d USING (store_id, product_id, dt)
        WHERE m.split = 'train' AND m.demand_m2 IS NOT NULL GROUP BY ALL ORDER BY ALL""").fetchnumpy()
    meta = {r[0]: r[1:] for r in con.execute(f"""
        SELECT product_id, count(DISTINCT store_id), avg((stockout_hours_open >= {threshold})::DOUBLE)
        FROM read_parquet('{daily}') WHERE split = 'train' GROUP BY ALL""").fetchall()}

    def take(mask, cols):
        return tuple(rows[c][mask] for c in cols)

    out = []
    for pid in np.unique(rows["product_id"]):
        tr = (rows["product_id"] == pid) & (rows["split"] == "train")
        ev = (rows["product_id"] == pid) & (rows["split"] == "eval")
        mm = m2rows["product_id"] == pid
        n_stores, rate = meta[int(pid)]
        out.append(ProductData(int(pid), int(n_stores), float(rate),
                               train=take(tr, ("y", "dow", "cens", "w")),
                               m2=tuple(m2rows[c][mm] for c in ("y", "dow", "w")),
                               heldout=take(ev, ("y", "dow", "cens", "w"))))
    return out


def fit_product(p: ProductData) -> list[dict] | str:
    """Fit all three methods for one product. Returns rows, or a reason string if a method lacks data."""
    y, dow, cens, w = p.train
    inputs = {
        "drop_stockout_days": (y[~cens], dow[~cens], np.zeros((~cens).sum(), bool), w[~cens]),
        "hourly_extrapolation": (p.m2[0], p.m2[1], np.zeros(len(p.m2[0]), bool), p.m2[2]),
        "censored_mle": (y, dow, cens, w),
    }
    for method, (_, _, _, ww) in inputs.items():
        if ww.sum() < MIN_OBS:
            return f"{method}: only {int(ww.sum())} training days"
    rows = []
    for method, args in inputs.items():
        f = cnb.fit(*args)
        held = cnb.heldout_loglik(f, *p.heldout) if p.heldout[3].sum() > 0 else float("nan")
        rows.append({"sku_id": p.product_id, "estimation_method": method, "lam_counts": f.lam,
                     **{f"dow_factor_{d}": float(f.dow[d]) for d in range(7)}, "dispersion_r": f.r,
                     "n_obs": f.n_obs, "n_censored": f.n_censored, "converged": f.converged, "loglik": f.loglik,
                     "heldout_loglik_per_day": held, "n_stores": p.n_stores, "stockout_day_rate": p.stockout_day_rate})
    return rows


def validate(table: pa.Table) -> list[str]:
    """C1 contract checks on the finished table. Returns problems (empty = ok)."""
    con = duckdb.connect()
    con.register("g", table)

    def q(sql: str):
        return con.execute(sql).fetchall()[0][0]

    dow_cols = [f"dow_factor_{d}" for d in range(7)]
    checks = {
        "unique (sku_id, estimation_method)": q("SELECT count(*) - count(DISTINCT (sku_id, estimation_method)) "
                                                "FROM g") == 0,
        "every product has all three methods": q("SELECT count(*) FROM (SELECT sku_id FROM g GROUP BY sku_id "
                                                 f"HAVING count(*) <> {len(METHODS)})") == 0,
        "lambda_base finite and > 0": q("SELECT count(*) FROM g WHERE NOT isfinite(lambda_base) "
                                        "OR lambda_base <= 0") == 0,
        "dow factors > 0 with mean 1": q(f"SELECT count(*) FROM g WHERE least({', '.join(dow_cols)}) <= 0 OR "
                                         f"abs(({' + '.join(dow_cols)}) / 7 - 1) > 1e-9") == 0,
        "0 <= n_censored <= n_obs": q("SELECT count(*) FROM g WHERE n_censored < 0 OR n_censored > n_obs") == 0,
        "method 1 uses no censored days": q("SELECT count(*) FROM g WHERE estimation_method = 'drop_stockout_days' "
                                            "AND n_censored > 0") == 0,
        f"non-converged share <= {MAX_NOT_CONVERGED}": q("SELECT coalesce(avg((NOT converged)::DOUBLE), 0) FROM g")
        <= MAX_NOT_CONVERGED,
        "at least one product": q("SELECT count(*) FROM g") > 0,
        "known methods only": q(f"SELECT count(*) FROM g WHERE estimation_method NOT IN {METHODS}") == 0,
    }
    return [name for name, ok in checks.items() if not ok]


def summarize(table: pa.Table) -> dict:
    """Headline numbers for question 4: how far apart are the methods, and which predicts the eval week best."""
    con = duckdb.connect()
    con.register("g", table)
    wide = """SELECT sku_id,
                     max(lambda_base) FILTER (WHERE estimation_method = 'drop_stockout_days') AS m1,
                     max(lambda_base) FILTER (WHERE estimation_method = 'hourly_extrapolation') AS m2,
                     max(lambda_base) FILTER (WHERE estimation_method = 'censored_mle') AS m3,
                     max(heldout_loglik_per_day) FILTER (WHERE estimation_method = 'drop_stockout_days') AS h1,
                     max(heldout_loglik_per_day) FILTER (WHERE estimation_method = 'hourly_extrapolation') AS h2,
                     max(heldout_loglik_per_day) FILTER (WHERE estimation_method = 'censored_mle') AS h3
              FROM g GROUP BY sku_id"""
    r = con.execute(f"""SELECT count(*), median(m3 / m1), median(m2 / m1), median(m3 / m2),
                               avg((h3 > h1)::DOUBLE), avg((h3 > h2)::DOUBLE), avg(h1), avg(h2), avg(h3)
                        FROM ({wide})""").fetchall()[0]
    keys = ["products", "median_lambda_ratio_m3_over_m1", "median_lambda_ratio_m2_over_m1",
            "median_lambda_ratio_m3_over_m2", "share_products_m3_beats_m1_heldout",
            "share_products_m3_beats_m2_heldout", "mean_heldout_ll_m1", "mean_heldout_ll_m2", "mean_heldout_ll_m3"]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in zip(keys, r, strict=True)}


def build(data_root: str | Path, source: Source | None = None, *, censor_threshold: int = 1,
          unit_multiplier: float = 10.0, workers: int = 4) -> Path:
    source = source or Source.load()
    data_root = Path(data_root)
    silver_dir = data_root / "silver" / source.revision
    silver_key = json.loads((silver_dir / "_manifest.json").read_text("utf-8"))["input_key"]
    params = {"censor_threshold": censor_threshold, "unit_multiplier": unit_multiplier, "min_obs": MIN_OBS}
    input_key = hashlib.sha256(json.dumps({"silver": silver_key, "gold_version": GOLD_VERSION, **params},
                                          sort_keys=True).encode()).hexdigest()[:16]
    final = data_root / "gold" / source.revision
    if (final / "_manifest.json").exists() and \
            json.loads((final / "_manifest.json").read_text("utf-8")).get("input_key") == input_key:
        log.info("gold up to date (input_key %s), skipping", input_key)
        return final

    t0 = time.perf_counter()
    products = load(silver_dir, censor_threshold, unit_multiplier)
    t_load = time.perf_counter() - t0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fit_product, products, chunksize=8))
    else:
        results = [fit_product(p) for p in products]
    skipped = {p.product_id: r for p, r in zip(products, results, strict=True) if isinstance(r, str)}
    rows = [row for r in results if not isinstance(r, str) for row in r]
    if not rows:
        sample = "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:5])
        raise GoldCheckError(f"no product could be estimated ({len(skipped)} skipped, e.g. {sample})")
    for row in rows:
        row["lambda_base"] = row.pop("lam_counts") / unit_multiplier
        row |= {"source": f"{source.dataset}@{source.revision[:12]}", "censor_threshold_hours": censor_threshold,
                "unit_multiplier": unit_multiplier, "schema_version": SCHEMA_VERSION}
    table = pa.Table.from_pylist(rows, schema=SCHEMA).sort_by([("sku_id", "ascending"),
                                                               ("estimation_method", "ascending")])
    problems = validate(table)
    if problems:
        raise GoldCheckError("C1 contract checks failed, nothing published: " + "; ".join(problems))

    staging = new_staging(data_root / "gold", source.revision[:12])
    pq.write_table(table, staging / "demand_params.parquet", compression="zstd")
    if validate(pq.read_table(staging / "demand_params.parquet")):  # audit what was actually written
        raise GoldCheckError("written file fails the contract")
    manifest = {"layer": "gold", "input_key": input_key, "gold_version": GOLD_VERSION, "revision": source.revision,
                "silver_input_key": silver_key, "params": params, "built_at": datetime.now(UTC).isoformat(),
                "seconds": {"load": round(t_load, 1), "total": round(time.perf_counter() - t0, 1)},
                "rows": table.num_rows, "skipped_products": skipped, "summary": summarize(table)}
    (staging / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    publish_dir(staging, final)
    log.info("gold published at %s: %d products, %d skipped", final, manifest["summary"]["products"], len(skipped))
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate per-product demand parameters (methods 1, 2, 3).")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--censor-threshold", type=int, default=1, help="stockout hours (6-21) that make a day censored")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        out = build(args.data_root, censor_threshold=args.censor_threshold, workers=args.workers)
    except GoldCheckError as exc:
        log.error("%s", exc)
        return 2
    print(f"gold ready at {out}")
    print(json.dumps(json.loads((out / "_manifest.json").read_text())["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
