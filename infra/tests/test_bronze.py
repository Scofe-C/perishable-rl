"""Bronze: schema contract, every check rejects bad data, typed output, idempotent rebuilds."""

import hashlib
import json

import duckdb
import pyarrow.parquet as pq
import pytest

from frn import bronze
from frn.bronze import BronzeCheckError, build
from frn.ingest import Source

REV = "f" * 40

# Literals are cast explicitly: DuckDB reads 0.1 as DECIMAL, which the schema contract (rightly) rejects.
# 3 stores x 2 products; train = 6 days, eval = the next 2 days. Shop open 6..19, so status 1 for hours <6 and >=20
# (=> 2 stockout hours inside 6..21). Sales only in hours 8..10 while in stock.
RAW_SQL = """
SELECT 1::BIGINT AS city_id, s::BIGINT AS store_id, 1::BIGINT AS management_group_id, 2::BIGINT AS first_category_id,
       3::BIGINT AS second_category_id, 4::BIGINT AS third_category_id, p::BIGINT AS product_id,
       strftime(DATE '2024-03-28' + d::INTEGER, '%Y-%m-%d') AS dt,
       list_sum(hs) AS sale_amount, hs AS hours_sale,
       list_sum(st[7:22])::INTEGER AS stock_hour6_22_cnt, st AS hours_stock_status,
       1.0::DOUBLE AS discount, 0::INTEGER AS holiday_flag, 0::INTEGER AS activity_flag,
       0.0::DOUBLE AS precpt, 20.0::DOUBLE AS avg_temperature, 60.0::DOUBLE AS avg_humidity,
       2.0::DOUBLE AS avg_wind_level
FROM range(3) a(s), range(2) b(p), range({d0}, {d1}) c(d),
     (SELECT list_transform(range(24), h -> CASE WHEN h BETWEEN 8 AND 10 THEN 0.1 ELSE 0.0 END)::DOUBLE[] AS hs,
             list_transform(range(24), h -> CASE WHEN h < 6 OR h >= 20 THEN 1 ELSE 0 END)::BIGINT[] AS st)
"""


def make_raw(root, mutate: dict[str, str] | None = None, train_days: int = 6) -> Source:
    """Write raw/<REV>/data/{train,eval}.parquet + _manifest.json. `mutate`: split -> SQL run on table t first."""
    raw = root / "raw" / REV / "data"
    raw.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    files = {}
    for split, (d0, d1) in {"train": (0, train_days), "eval": (train_days, train_days + 2)}.items():
        con.execute(f"CREATE OR REPLACE TABLE t AS {RAW_SQL.format(d0=d0, d1=d1)}")
        if mutate and split in mutate:
            con.execute(mutate[split])
        f = raw / f"{split}.parquet"
        con.execute(f"COPY t TO '{f.as_posix()}' (FORMAT parquet)")
        files[f"data/{split}.parquet"] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": 1}
    (raw.parent / "_manifest.json").write_text(json.dumps({"revision": REV, "files": files}))
    return Source("org/ds", REV, "file:///unused", files)


def out_files(root):
    return sorted(p.name for p in (root / "bronze" / REV).glob("*")) if (root / "bronze" / REV).exists() else []


def test_happy_path_types_rows_and_manifest(tmp_path):
    out = build(tmp_path, make_raw(tmp_path))
    con = duckdb.connect()
    types = dict(con.execute(f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM "
                             f"read_parquet('{(out / 'train.parquet').as_posix()}'))").fetchall())
    assert types["dt"] == "DATE" and types["store_id"] == "INTEGER" and types["holiday_flag"] == "BOOLEAN"
    assert types["hours_stock_status"] == "TINYINT[]" and types["sale_amount"] == "DOUBLE"
    assert pq.read_metadata(out / "train.parquet").num_rows == 3 * 2 * 6
    keys = con.execute(f"SELECT store_id, product_id, dt FROM read_parquet('{(out / 'train.parquet').as_posix()}')"
                       ).fetchall()
    assert keys == sorted(keys)
    m = json.loads((out / "_manifest.json").read_text())
    assert m["stats"]["eval"]["rows"] == 12 and m["stats"]["train"]["date_min"] == "2024-03-28"
    assert all(c["passed"] for split in m["checks"].values() for c in split)
    assert not list(out.rglob("*.tmp"))


def test_rebuild_is_skipped_unless_inputs_or_version_change(tmp_path, monkeypatch):
    src = make_raw(tmp_path)
    out = build(tmp_path, src)
    mtime = (out / "train.parquet").stat().st_mtime_ns
    build(tmp_path, src)
    assert (out / "train.parquet").stat().st_mtime_ns == mtime          # skipped
    monkeypatch.setattr(bronze, "BRONZE_VERSION", bronze.BRONZE_VERSION + 1)
    build(tmp_path, src)
    assert (out / "train.parquet").stat().st_mtime_ns != mtime          # rebuilt


@pytest.mark.parametrize("mutate, match", [
    ("ALTER TABLE t ADD COLUMN price DOUBLE DEFAULT 1.0", "unexpected column price"),
    ("ALTER TABLE t RENAME COLUMN discount TO disc", "missing column discount"),
    ("ALTER TABLE t ALTER sale_amount TYPE FLOAT", "sale_amount: type FLOAT"),
])
def test_schema_drift_stops_everything(tmp_path, mutate, match):
    with pytest.raises(BronzeCheckError, match=match):
        build(tmp_path, make_raw(tmp_path, {"train": mutate}))
    assert out_files(tmp_path) == []


@pytest.mark.parametrize("mutate, check", [
    ("INSERT INTO t SELECT * FROM t LIMIT 1", "unique key"),
    ("DELETE FROM t WHERE store_id = 0 AND product_id = 0 AND dt = '2024-03-28'", "complete panel"),
    ("UPDATE t SET precpt = NULL WHERE store_id = 1", "no nulls"),
    ("UPDATE t SET dt = '2024/03/30' WHERE dt = '2024-03-30'", "dt parses"),
    ("UPDATE t SET hours_stock_status = list_transform(hours_stock_status, x -> x * 2)", "stock status is 0/1"),
    ("UPDATE t SET activity_flag = 3 WHERE store_id = 2", "flags are 0/1"),
    ("UPDATE t SET sale_amount = sale_amount + 0.5 WHERE store_id = 1", "sale_amount == sum(hours_sale)"),
    ("UPDATE t SET stock_hour6_22_cnt = stock_hour6_22_cnt + 1 WHERE store_id = 0", "stock_hour6_22_cnt =="),
    (("UPDATE t SET hours_stock_status = list_transform(hours_stock_status, x -> 1 - x), "
      "stock_hour6_22_cnt = 14"), "status 1 means out of stock"),
    ("UPDATE t SET discount = 2.5 WHERE store_id = 0", "discount in [0, 2]"),
    ("UPDATE t SET hours_sale = list_transform(hours_sale, x -> -x), sale_amount = -sale_amount", "sales >= 0"),
])
def test_each_check_rejects_bad_data_and_writes_nothing(tmp_path, mutate, check):
    with pytest.raises(BronzeCheckError, match="nothing written") as exc:
        build(tmp_path, make_raw(tmp_path, {"train": mutate}))
    assert check in str(exc.value)
    assert out_files(tmp_path) == []


def test_eval_overlapping_train_is_rejected(tmp_path):
    mutate = {"eval": "UPDATE t SET dt = strftime(strptime(dt, '%Y-%m-%d') - INTERVAL 5 DAY, '%Y-%m-%d')"}
    with pytest.raises(BronzeCheckError, match="strictly after train"):
        build(tmp_path, make_raw(tmp_path, mutate))


def test_raw_manifest_for_another_revision_is_rejected(tmp_path):
    src = make_raw(tmp_path)
    other = Source(src.dataset, "0" * 40, src.base_url, src.files)
    raw_dir = tmp_path / "raw" / ("0" * 40)
    raw_dir.mkdir(parents=True)
    (raw_dir / "_manifest.json").write_text(json.dumps({"revision": REV, "files": src.files}))
    with pytest.raises(BronzeCheckError, match="raw manifest is for"):
        build(tmp_path, other)


def test_warnings_are_recorded_but_do_not_block(tmp_path):
    """Real-data case: a few discounts slightly above 1 and some zeros. Built anyway, recorded in the manifest."""
    mutate = {"train": "UPDATE t SET discount = 1.088 WHERE store_id = 0 AND product_id = 0; "
                       "UPDATE t SET discount = 0.0 WHERE store_id = 1 AND product_id = 1"}
    out = build(tmp_path, make_raw(tmp_path, mutate))
    m = json.loads((out / "_manifest.json").read_text())
    by_name = {c["name"]: c for c in m["checks"]["train"]}
    assert by_name["discount in [0, 1]"] == {"name": "discount in [0, 1]", "value": 6, "passed": False,
                                            "severity": "warn"}
    assert by_name["discount == 0 rows (likely missing, not free)"]["value"] == 6
    assert by_name["discount in [0, 2]"]["passed"]
    assert pq.read_metadata(out / "train.parquet").num_rows == 36       # rows kept, values unchanged
