"""Silver (dbt) write-audit-publish: publish only after every dbt test passes; failures leave no trace."""

import json

import duckdb
import pytest
from test_bronze import make_raw

from frn import bronze, silver
from frn.silver import SilverBuildError


def silver_dirs(root):
    return sorted(p.name for p in (root / "silver").iterdir() if p.name != "_dbt_runs")


@pytest.fixture
def built(tmp_path):
    src = make_raw(tmp_path)
    bronze.build(tmp_path, src)
    return tmp_path, src


def test_build_publishes_tested_outputs_then_skips(built):
    root, src = built
    out = silver.build(root, src)
    m = json.loads((out / "_manifest.json").read_text())
    assert m["tests"]["total"] >= 20 and m["tests"]["passed"] == m["tests"]["total"]
    assert silver_dirs(root) == [src.revision]                      # staging folder is gone
    assert not (out / "_build.duckdb").exists() and (out / "_dbt_run_results.json").exists()
    rows = duckdb.sql(f"SELECT count(*) FROM read_parquet('{(out / 'daily.parquet').as_posix()}')").fetchall()
    assert rows == [(36 + 12,)]                                    # train + eval rows of the fixture
    # the fixture has a stockout in hours 20-21 every day, so no clean days: uniform profile, still one row per day
    prof = duckdb.sql(f"SELECT DISTINCT level, share FROM read_parquet('{(out / 'hour_profile.parquet').as_posix()}')")
    assert prof.fetchall() == [("uniform", 1 / 16)]
    built_at = m["built_at"]
    silver.build(root, src)                                        # nothing changed -> skipped
    assert json.loads((out / "_manifest.json").read_text())["built_at"] == built_at


def test_failing_dbt_test_publishes_nothing(tmp_path):
    """A product in two categories breaks the category fallback; the singular dbt test must stop the publish."""
    src = make_raw(tmp_path, {"train": "UPDATE t SET third_category_id = 99 WHERE product_id = 0 AND store_id = 0"})
    bronze.build(tmp_path, src)                                    # bronze has no opinion on categories
    with pytest.raises(SilverBuildError, match="one_category_per_product"):
        silver.build(tmp_path, src)
    assert silver_dirs(tmp_path) == []                             # no product, no staging leftovers


def test_failure_keeps_the_previous_version(built, monkeypatch):
    root, src = built
    out = silver.build(root, src)
    before = (out / "_manifest.json").read_text()

    def boom(args):
        raise SilverBuildError("simulated failing data test")

    monkeypatch.setattr(silver, "_run_dbt", boom)
    with pytest.raises(SilverBuildError):
        silver.build(root, src, dbt_vars={"max_censored_share": 0.5})   # new vars -> a rebuild is attempted
    assert (out / "_manifest.json").read_text() == before
    assert silver_dirs(root) == [src.revision]


def test_project_fingerprint_ignores_line_endings(tmp_path):
    for eol, name in ((b"\n", "a"), (b"\r\n", "b")):
        d = tmp_path / name / "models"
        d.mkdir(parents=True)
        (d / "m.sql").write_bytes(b"select 1" + eol + b"from x" + eol)
    assert silver.project_fingerprint(tmp_path / "a") == silver.project_fingerprint(tmp_path / "b")
