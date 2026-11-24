"""Gold end to end on a synthetic silver layer with a planted truth: recovery, contract, publish, skip."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from frn import gold
from frn.gold import GoldCheckError
from frn.ingest import Source

REV = "e" * 40
SRC = Source("org/ds", REV, "file:///unused", {})
TRUE_LAM = {1: 0.8, 2: 1.5}          # normalized units (counts / 10)
R = 4.0


def make_silver(root, stock_mean=10.0, seed=0):
    """Silver daily + demand_m2 for 2 products x 15 stores, 90 train + 7 eval days. Sales are censored by a
    random stock level; a sold-out day gets 4 stockout hours. demand_m2 is an oracle (true demand)."""
    rng = np.random.default_rng(seed)
    dates = np.arange(np.datetime64("2024-03-28"), np.datetime64("2024-07-03"))
    rows = []
    for pid, lam in TRUE_LAM.items():
        for store in range(15):
            for i, dt in enumerate(dates):
                mu = lam * 10
                demand = rng.negative_binomial(R, R / (R + mu))
                stock = rng.poisson(stock_mean * lam)
                sold = min(demand, stock)
                cens = demand >= stock
                rows.append({"store_id": store, "product_id": pid, "dt": dt.astype("datetime64[D]").item(),
                             "split": "train" if i < 90 else "eval", "dow": int((dt.astype(int) + 3) % 7),
                             "sales": sold / 10, "stockout_hours_open": 4 if cens else 0, "demand_m2": demand / 10})
    d = root / "silver" / REV
    d.mkdir(parents=True)
    t = pa.Table.from_pylist(rows)
    pq.write_table(t.drop(["demand_m2"]), d / "daily.parquet")
    pq.write_table(t.select(["store_id", "product_id", "dt", "split", "demand_m2"]), d / "demand_m2.parquet")
    (d / "_manifest.json").write_text(json.dumps({"input_key": f"silver-{seed}-{stock_mean}"}))
    return d


def params(out) -> list[dict]:
    return pq.read_table(out / "demand_params.parquet").to_pylist()


def by_method(rows, pid):
    return {r["estimation_method"]: r for r in rows if r["sku_id"] == pid}


def test_recovers_planted_lambda_and_method1_is_biased(tmp_path):
    make_silver(tmp_path)
    out = gold.build(tmp_path, SRC, workers=1)
    rows = params(out)
    assert len(rows) == 2 * 3 and {r["estimation_method"] for r in rows} == set(gold.METHODS)
    for pid, lam in TRUE_LAM.items():
        by = by_method(rows, pid)
        assert by["censored_mle"]["lambda_base"] == pytest.approx(lam, rel=0.06)
        assert by["hourly_extrapolation"]["lambda_base"] == pytest.approx(lam, rel=0.06)   # oracle recovery
        assert by["drop_stockout_days"]["lambda_base"] < 0.9 * lam                        # biased low
        assert by["drop_stockout_days"]["n_censored"] == 0 and by["censored_mle"]["n_censored"] > 0
    m = json.loads((out / "_manifest.json").read_text())
    assert m["summary"]["median_lambda_ratio_m3_over_m1"] > 1.1
    assert m["summary"]["share_products_m3_beats_m1_heldout"] == 1.0
    dow = np.array([[r[f"dow_factor_{d}"] for d in range(7)] for r in rows])
    np.testing.assert_allclose(dow.mean(axis=1), 1.0, atol=1e-9)


def test_skip_when_unchanged_and_rebuild_when_threshold_changes(tmp_path):
    make_silver(tmp_path)
    out = gold.build(tmp_path, SRC, workers=1)
    t = json.loads((out / "_manifest.json").read_text())["built_at"]
    gold.build(tmp_path, SRC, workers=1)
    assert json.loads((out / "_manifest.json").read_text())["built_at"] == t
    gold.build(tmp_path, SRC, workers=1, censor_threshold=5)     # 4 stockout hours < 5: nothing censored now
    m = json.loads((out / "_manifest.json").read_text())
    assert m["built_at"] != t and m["params"]["censor_threshold"] == 5
    assert all(r["n_censored"] == 0 for r in params(out))


def test_contract_violation_publishes_nothing(tmp_path, monkeypatch):
    make_silver(tmp_path)
    real_fit = gold.cnb.fit

    def broken_fit(*a, **k):
        f = real_fit(*a, **k)
        return f.__class__(**{**f.__dict__, "dow": f.dow * 1.5})     # dow factors no longer mean 1

    monkeypatch.setattr(gold.cnb, "fit", broken_fit)
    with pytest.raises(GoldCheckError, match="dow factors > 0 with mean 1"):
        gold.build(tmp_path, SRC, workers=1)
    assert not (tmp_path / "gold").exists() or list((tmp_path / "gold").iterdir()) == []


def test_no_estimable_product_is_an_error_that_says_why(tmp_path):
    make_silver(tmp_path, stock_mean=0.3)          # almost always sold out -> method 1 has too few days
    with pytest.raises(GoldCheckError, match="no product could be estimated.*drop_stockout_days: only"):
        gold.build(tmp_path, SRC, workers=1)
    assert not (tmp_path / "gold").exists()
