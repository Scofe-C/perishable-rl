"""Validate an EvalResult and write it as a fact_step Parquet file. Contract: docs/WAREHOUSE.md §3 and §6.

Nothing is written unless every check passes, and the write is atomic (tmp + rename).

Owner: Member 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from perinv.env.config import EnvConfig
from perinv.eval.evaluate import QTY_FIELDS, EvalResult
from perinv.utils.atomic import atomic_path

EVAL_SCHEMA_VERSION = 1
REWARD_TOL = 1e-6
_INT16_MAX = np.iinfo(np.int16).max

SCHEMA = pa.schema([
    ("run_key", pa.dictionary(pa.int32(), pa.string())),
    ("path_id", pa.int16()),
    ("t", pa.int16()),
    ("sku", pa.int16()),
    *[(f, pa.int32()) for f in QTY_FIELDS],
    ("day_reward", pa.float64()),  # sku = 0 row only, NULL elsewhere
    ("clipped", pa.bool_()),       # sku = 0 row only, NULL elsewhere
])


class EvalValidationError(ValueError):
    """The evaluation result breaks a WAREHOUSE.md §6 check; nothing was written."""


def sku_rewards(result: EvalResult, cfg: EnvConfig) -> np.ndarray:
    """float64[P, T, N]: each SKU's reward recomputed from quantities and unit costs (SPEC §3 step 7)."""
    q = result.qty

    def c(name: str) -> np.ndarray:  # per-SKU cost vector, broadcast over [P, T, N]
        return np.asarray(getattr(cfg, name), dtype=np.float64)

    return (c("price") * q["sold"] - c("order_cost") * q["order"] - c("holding_cost") * q["held"]
            - c("expiry_cost") * q["expired"] - c("lost_sales_cost") * q["lost"])


def validate(result: EvalResult, cfg: EnvConfig, demand: np.ndarray) -> None:
    """Raise EvalValidationError on the first broken check (WAREHOUSE.md §6)."""
    p_count, horizon, n = result.shape
    ids = result.path_ids

    def fail(msg: str) -> None:
        raise EvalValidationError(msg)

    if (horizon, n) != (cfg.horizon, cfg.n_skus):
        fail(f"result shape (T={horizon}, N={n}) does not match config (T={cfg.horizon}, N={cfg.n_skus})")
    for f in QTY_FIELDS:
        if result.qty[f].shape != (p_count, horizon, n):
            fail(f"{f} has shape {result.qty[f].shape}, expected {(p_count, horizon, n)}")
    if result.day_reward.shape != (p_count, horizon) or result.clipped.shape != (p_count, horizon):
        fail("day_reward / clipped must be [P, T]")
    if len(np.unique(ids)) != p_count:
        fail("duplicate path_id: the grain (run_key, path_id, t, sku) must be unique")
    if ids.min(initial=0) < 0 or ids.max(initial=0) > _INT16_MAX or horizon > _INT16_MAX:
        fail("path_id and t must fit in int16")

    for f in QTY_FIELDS:
        if (result.qty[f] < 0).any():
            fail(f"negative {f}")
    if not np.array_equal(result.qty["sold"] + result.qty["lost"], np.asarray(demand)[ids]):
        fail("sold + lost does not equal the CRN demand")
    used = (result.qty["occ"] + result.qty["order"]).sum(axis=2)
    if (used > cfg.capacity).any():
        p, t = np.argwhere(used > cfg.capacity)[0]
        fail(f"capacity exceeded on path {ids[p]}, day {t}: {used[p, t]} > {cfg.capacity}")
    bad = ~np.isfinite(result.day_reward)
    if bad.any():   # must come first: NaN > tol is False, so the reconcile check below would let NaN through
        p, t = np.argwhere(bad)[0]
        fail(f"non-finite reward on path {ids[p]}, day {t}: {result.day_reward[p, t]}")
    gap = np.abs(sku_rewards(result, cfg).sum(axis=2) - result.day_reward)
    if (gap > REWARD_TOL).any():
        p, t = np.argwhere(gap > REWARD_TOL)[0]
        fail(f"reward does not reconcile on path {ids[p]}, day {t}: recomputed differs by {gap[p, t]:.3g}")


def to_table(result: EvalResult, run_key: str) -> pa.Table:
    """Build the fact table straight from numpy (no pandas), sorted by (path_id, t, sku)."""
    order = np.argsort(result.path_ids, kind="stable")
    p_count, horizon, n = result.shape
    rows = p_count * horizon * n

    path = np.repeat(result.path_ids[order].astype(np.int16), horizon * n)
    t = np.tile(np.repeat(np.arange(horizon, dtype=np.int16), n), p_count)
    sku = np.tile(np.arange(n, dtype=np.int16), p_count * horizon)
    not_first_sku = sku != 0  # day-level values live on the sku = 0 row only

    cols = {
        "run_key": pa.DictionaryArray.from_arrays(pa.array(np.zeros(rows, np.int32)), pa.array([run_key])),
        "path_id": pa.array(path), "t": pa.array(t), "sku": pa.array(sku),
        **{f: pa.array(result.qty[f][order].reshape(-1)) for f in QTY_FIELDS},
        "day_reward": pa.array(np.repeat(result.day_reward[order].reshape(-1), n),
                               mask=not_first_sku, type=pa.float64()),
        "clipped": pa.array(np.repeat(result.clipped[order].reshape(-1), n), mask=not_first_sku, type=pa.bool_()),
    }
    meta = {
        "perinv.eval_schema_version": str(EVAL_SCHEMA_VERSION),
        "perinv.run_key": run_key,
        "perinv.n_paths": str(p_count), "perinv.T": str(horizon), "perinv.N": str(n),
    }
    return pa.Table.from_pydict(cols, schema=SCHEMA.with_metadata(meta))


def write_eval(result: EvalResult, cfg: EnvConfig, demand: np.ndarray, run_key: str, path: str | Path) -> Path:
    """Validate, then atomically write `path` (usually runs/<run_name>/eval_test.parquet)."""
    validate(result, cfg, demand)
    table = to_table(result, run_key)
    path = Path(path)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")
    return path
