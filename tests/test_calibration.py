"""Calibrated configs from a synthetic C1 file: selection rules, one global scale, shared capacity, loadable YAML."""

import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perinv.data import calibration as cal
from perinv.env.config import load_env_config

REPO = Path(__file__).parents[1]
# sku: (n_stores, stockout rate, lambda m1, lambda m3, converged)
SKUS = {
    10: (500, 0.45, 2.0, 5.0, True),     # top candidate -> scale = 4 / 5.0
    11: (400, 0.50, 1.0, 1.5, True),
    12: (300, 0.35, 0.8, 1.0, True),
    13: (250, 0.40, 0.5, 0.6, True),
    14: (200, 0.55, 0.9, 1.2, True),
    20: (900, 0.80, 1.0, 3.0, True),     # stockout rate too high -> excluded
    21: (800, 0.45, 1.0, 2.0, False),    # a fit did not converge -> excluded
    22: (50, 0.40, 1.0, 2.0, True),      # too few stores -> excluded
}
DOW = [0.9, 0.9, 0.9, 0.9, 1.0, 1.2, 1.2]


def write_c1(path, schema_version=1):
    rows = []
    for sku, (stores, rate, l1, l3, conv) in SKUS.items():
        for method, lam in (("drop_stockout_days", l1), ("hourly_extrapolation", l3), ("censored_mle", l3)):
            rows.append({"sku_id": sku, "source": "test@abc", "estimation_method": method, "lambda_base": lam,
                         **{f"dow_factor_{d}": DOW[d] for d in range(7)}, "converged": conv,
                         "n_stores": stores, "stockout_day_rate": rate, "schema_version": schema_version})
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_selection_rules_and_order(tmp_path):
    params = cal.load_c1(write_c1(tmp_path / "c1.parquet"))
    assert cal.select_skus(params, 5) == [10, 11, 12, 13, 14]
    with pytest.raises(ValueError, match="only 5 SKUs"):
        cal.select_skus(params, 6)


def test_one_global_scale_shared_capacity_and_nested_skus(tmp_path):
    configs = cal.build_configs(cal.load_c1(write_c1(tmp_path / "c1.parquet")))
    assert len(configs) == 3 * 2 * 2
    m3, m1 = configs["frn_m3_n1_k10"], configs["frn_m1_n1_k10"]
    assert m3["demand_mean"] == [4.0]                         # top SKU, censored MLE -> team default lambda
    assert m1["demand_mean"] == [pytest.approx(2.0 * 4.0 / 5.0)]
    assert m1["capacity"] == m3["capacity"] == 20             # 1.0 * 5 * 4
    assert configs["frn_m3_n1_k06"]["capacity"] == 12
    n5 = configs["frn_m3_n5_k06"]
    assert n5["calibration"]["sku_ids"] == [10, 11, 12, 13, 14]
    assert configs["frn_m3_n2_k06"]["calibration"]["sku_ids"] == [10, 11]
    assert n5["capacity"] == round(0.6 * 5 * sum(n5["demand_mean"]))
    assert sum(n5["season"]) / 7 == pytest.approx(1.0, abs=1e-5)


def test_cli_writes_loadable_env_configs(tmp_path):
    c1 = write_c1(tmp_path / "c1.parquet")
    out = tmp_path / "calibrated"
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "make_calibrated_configs.py"), str(c1),
                          "--out", str(out)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    files = sorted(out.glob("*.yaml"))
    assert len(files) == 12
    env = load_env_config(out / "frn_m1_n5_k06.yaml")
    assert env.n_skus == 5 and len(env.demand_mean) == 5 and env.kappa == 0.6
    assert env.demand_hash() != load_env_config(out / "frn_m3_n5_k06.yaml").demand_hash()   # own CRN paths


def test_refuses_non_c1_files(tmp_path):
    with pytest.raises(ValueError, match="schema v2"):
        cal.load_c1(write_c1(tmp_path / "v2.parquet", schema_version=2))
    pq.write_table(pa.table({"x": [1]}), tmp_path / "other.parquet")
    with pytest.raises(ValueError, match="not a C1"):
        cal.load_c1(tmp_path / "other.parquet")
