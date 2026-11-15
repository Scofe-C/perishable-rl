"""Dagster wiring: the four assets run in order, checks report, reruns are no-ops, a bad layer stops downstream."""


import yaml
from test_bronze import make_raw

import dagster as dg
from frn.definitions import FrnStorage, defs


def storage_for(tmp_path, mutate=None) -> FrnStorage:
    src = make_raw(tmp_path, mutate, train_days=15)   # 3 stores x 15 days = 45 days/product >= gold.MIN_OBS
    source_yaml = tmp_path / "source.yaml"
    source_yaml.write_text(yaml.safe_dump({"dataset": src.dataset, "revision": src.revision,
                                           "base_url": src.base_url, "files": src.files}))
    return FrnStorage(data_root=str(tmp_path), source_path=str(source_yaml))


# The fixture has stockouts in hours 20-21 every day; threshold 3 makes those days count as uncensored,
# so method 1 has data. workers=1 keeps the test in-process.
RUN_CONFIG = {"ops": {"demand_params": {"config": {"censor_threshold": 3, "workers": 1}}}}


def run(storage, raise_on_error=True):
    return dg.materialize(list(defs.assets) + list(defs.asset_checks), resources={"storage": storage},
                          run_config=RUN_CONFIG, raise_on_error=raise_on_error)


def test_definitions_load_and_job_covers_all_assets():
    job = defs.resolve_job_def("freshretailnet_pipeline")
    keys = {k.to_user_string() for k in job.asset_layer.selected_asset_keys}
    assert keys == {"raw", "bronze_tables", "silver_tables", "demand_params"}


def test_full_pipeline_materializes_with_passing_checks_then_rerun_is_noop(tmp_path):
    storage = storage_for(tmp_path)
    result = run(storage)
    assert result.success
    mats = {m.asset_key.to_user_string() for m in result.get_asset_materialization_events()}
    assert mats == {"raw", "bronze_tables", "silver_tables", "demand_params"}
    checks = {e.check_name: e.passed for e in result.get_asset_check_evaluations()}
    assert checks == {"bronze_checks_passed": True, "silver_dbt_tests_passed": True,
                      "demand_params_contract": True, "censoring_direction": checks["censoring_direction"]}
    built = {layer: storage.manifest(layer).get("built_at") for layer in ("bronze", "silver", "gold")}
    assert run(storage).success                                     # nothing changed: every layer skips
    assert {layer: storage.manifest(layer).get("built_at") for layer in ("bronze", "silver", "gold")} == built


def test_bad_bronze_data_stops_downstream_assets(tmp_path):
    storage = storage_for(tmp_path, {"train": "UPDATE t SET discount = 9.0 WHERE store_id = 0"})
    result = run(storage, raise_on_error=False)
    assert not result.success
    mats = {m.asset_key.to_user_string() for m in result.get_asset_materialization_events()}
    assert mats == {"raw"}                                          # bronze failed; silver and gold never ran
    assert not (tmp_path / "silver").exists() and not (tmp_path / "gold").exists()
