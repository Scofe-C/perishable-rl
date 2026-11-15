"""evaluate() + writer: contract checks from docs/WAREHOUSE.md §3 and §6."""

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest
from fakes import ConstantPolicy, FakeEnv, RandomPolicy, tiny_config

from perinv.data.crn import generate_paths
from perinv.eval import writer
from perinv.eval.evaluate import evaluate
from perinv.eval.writer import EVAL_SCHEMA_VERSION, SCHEMA, EvalValidationError, write_eval

N_PATHS = 6


@pytest.fixture
def setup():
    cfg = tiny_config()
    return cfg, FakeEnv(cfg), generate_paths(cfg, "test", N_PATHS, root_seed=7)


def test_evaluate_shapes_and_per_path_profit(setup):
    cfg, env, demand = setup
    res = evaluate(RandomPolicy(cfg.n_skus), env, demand)
    assert res.shape == (N_PATHS, cfg.horizon, cfg.n_skus)
    df = res.per_path()
    assert list(df.path_id) == list(range(N_PATHS))
    np.testing.assert_allclose(df.profit, res.day_reward.sum(axis=1))
    assert (df.total_demand == demand.sum(axis=(1, 2))).all()
    assert res.clipped.any(), "RandomPolicy should trigger clipping at least once (exercises the flag)"


def test_shuffled_path_order_gives_identical_per_path_results(setup):
    """Replaces the M3 'eval invariance' pending guard: each path is seeded by its own index."""
    cfg, env, demand = setup
    a = evaluate(RandomPolicy(cfg.n_skus), env, demand).per_path()
    b = evaluate(RandomPolicy(cfg.n_skus), env, demand, path_ids=[4, 1, 5, 0, 3, 2]).per_path()
    assert a.equals(b)
    # a subset gives the same rows as the full run
    c = evaluate(RandomPolicy(cfg.n_skus), env, demand, path_ids=[3, 1]).per_path()
    assert c.equals(a[a.path_id.isin([1, 3])].reset_index(drop=True))


def test_written_file_matches_contract(setup, tmp_path):
    cfg, env, demand = setup
    res = evaluate(RandomPolicy(cfg.n_skus), env, demand, path_ids=[2, 0, 1, 5, 4, 3])
    out = write_eval(res, cfg, demand, "abc123", tmp_path / "eval_test.parquet")
    assert not list(tmp_path.glob("*.tmp"))

    table = pq.read_table(out)
    assert table.schema.remove_metadata().equals(SCHEMA)
    meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert meta["perinv.eval_schema_version"] == str(EVAL_SCHEMA_VERSION)
    assert meta["perinv.run_key"] == "abc123"
    assert table.num_rows == N_PATHS * cfg.horizon * cfg.n_skus

    con = duckdb.connect()
    con.execute(f"CREATE VIEW f AS SELECT * FROM read_parquet('{out.as_posix()}')")
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    # grain is unique and rows are sorted by (path_id, t, sku)
    assert q("SELECT count(*) = count(DISTINCT (run_key, path_id, t, sku)) FROM f") == [(True,)]
    keys = q("SELECT path_id, t, sku FROM f")
    assert keys == sorted(keys)
    # day-level columns are on the sku = 0 row only, so a plain SUM is the true total
    assert q("SELECT count(*) FROM f WHERE sku <> 0 AND (day_reward IS NOT NULL OR clipped IS NOT NULL)") == [(0,)]
    total = q("SELECT sum(day_reward) FROM f")[0][0]
    assert total == pytest.approx(res.day_reward.sum())
    assert q("SELECT sum(sold + lost) FROM f")[0][0] == demand.sum()


def _bad(setup):
    cfg, env, demand = setup
    return cfg, demand, evaluate(ConstantPolicy([3, 2]), env, demand)


@pytest.mark.parametrize("corrupt, match", [
    (lambda r: r.qty["held"].__setitem__((0, 0, 0), -1), "negative held"),
    (lambda r: r.qty["sold"].__setitem__((1, 2, 1), r.qty["sold"][1, 2, 1] + 1), "CRN demand"),
    (lambda r: r.qty["order"].__setitem__((0, 3, 0), 999), "capacity exceeded"),
    (lambda r: r.day_reward.__setitem__((2, 4), r.day_reward[2, 4] + 0.01), "reconcile"),
    (lambda r: r.path_ids.__setitem__(1, 0), "duplicate path_id"),
])
def test_writer_rejects_bad_results_and_writes_nothing(setup, tmp_path, corrupt, match):
    cfg, demand, res = _bad(setup)
    corrupt(res)
    with pytest.raises(EvalValidationError, match=match):
        write_eval(res, cfg, demand, "k", tmp_path / "eval_test.parquet")
    assert list(tmp_path.iterdir()) == []


def test_crash_during_write_leaves_no_file(setup, tmp_path, monkeypatch):
    cfg, demand, res = _bad(setup)

    def boom(table, where, **kw):
        open(where, "wb").write(b"half a parquet file")
        raise OSError("disk full")

    monkeypatch.setattr(writer.pq, "write_table", boom)
    with pytest.raises(OSError, match="disk full"):
        write_eval(res, cfg, demand, "k", tmp_path / "eval_test.parquet")
    assert list(tmp_path.iterdir()) == []


def test_evaluate_rejects_early_termination(setup):
    cfg, env, demand = setup

    class StopsEarly(FakeEnv):
        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            return obs, r, self.t == 3, trunc, info

    with pytest.raises(RuntimeError, match="truncates exactly at t=T"):
        evaluate(ConstantPolicy([1, 1]), StopsEarly(cfg), demand)
