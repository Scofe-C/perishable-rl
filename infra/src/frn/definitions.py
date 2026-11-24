"""Dagster orchestration: raw -> bronze -> silver -> gold as four assets, plus asset checks.

    dagster dev -m frn.definitions                                        # UI at http://localhost:3000
    dagster asset materialize -m frn.definitions --select '*'            # headless run

Each asset calls its layer's own build() function, so the guarantees live in one place and hold with or
without Dagster: pinned + verified inputs, all-or-nothing writes, write-audit-publish for dbt, and "skip if
nothing changed". Dagster adds dependency order, lineage, run history, retries and a UI on top.

Why not one Dagster asset per dbt model (dagster-dbt)? It would run `dbt build` straight into the published
folder and bypass the audit-before-publish step. dbt test results show up as asset checks instead.

No schedule: the source is a pinned, static dataset. A run when nothing changed is a fast no-op anyway.
"""

# No `from __future__ import annotations` here: Dagster reads the annotations at runtime to find resources
# and config classes, and string annotations ("GoldConfig") cannot be resolved.
import json
import os
from pathlib import Path

import dagster as dg
from frn import bronze, gold, silver
from frn.ingest import DEFAULT_SOURCE, Source, ingest


class FrnStorage(dg.ConfigurableResource):
    """Where the data lives and which source pin to use."""

    data_root: str = os.environ.get("FRN_DATA_ROOT", "/data/freshretailnet")
    source_path: str = str(DEFAULT_SOURCE)

    def source(self) -> Source:
        return Source.load(self.source_path)

    def layer_dir(self, layer: str) -> Path:
        return Path(self.data_root) / layer / self.source().revision

    def manifest(self, layer: str) -> dict:
        name = "_manifest.json"
        return json.loads((self.layer_dir(layer) / name).read_text(encoding="utf-8"))


class GoldConfig(dg.Config):
    censor_threshold: int = 1   # stockout hours (6..21) that make a day censored
    workers: int = 4


# ---------------------------------------------------------------- assets

@dg.asset(group_name="freshretailnet", compute_kind="python",
          retry_policy=dg.RetryPolicy(max_retries=3, delay=10, backoff=dg.Backoff.EXPONENTIAL),
          description="Pinned FreshRetailNet revision, every file sha256-verified (frn.ingest).")
def raw(storage: FrnStorage) -> dg.MaterializeResult:
    out = ingest(storage.data_root, storage.source())
    m = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    return dg.MaterializeResult(metadata={
        "revision": m["revision"],
        "files": dg.MetadataValue.json(
            {k: {"bytes": v["bytes"], "action": v["action"]} for k, v in m["files"].items()}),
        "downloaded": sum(v["action"] == "downloaded" for v in m["files"].values()),
    })


@dg.asset(deps=[raw], group_name="freshretailnet", compute_kind="duckdb",
          description="Typed Parquet; schema contract + data checks, nothing written on failure (frn.bronze).")
def bronze_tables(storage: FrnStorage) -> dg.MaterializeResult:
    bronze.build(storage.data_root, storage.source())
    m = storage.manifest("bronze")
    return dg.MaterializeResult(metadata={
        "rows_train": m["stats"]["train"]["rows"], "rows_eval": m["stats"]["eval"]["rows"],
        "mb": round(sum(s["bytes"] for s in m["stats"].values()) / 1e6, 1), "bronze_version": m["bronze_version"],
    })


@dg.asset(deps=[bronze_tables], group_name="freshretailnet", compute_kind="dbt",
          description="dbt models + tests, built in staging and published only if all tests pass (frn.silver).")
def silver_tables(storage: FrnStorage) -> dg.MaterializeResult:
    silver.build(storage.data_root, storage.source())
    m = storage.manifest("silver")
    return dg.MaterializeResult(metadata={
        "dbt_tests": f"{m['tests']['passed']}/{m['tests']['total']}",
        "outputs": dg.MetadataValue.json(m["outputs"]), "dbt_project_fingerprint": m["dbt_project_fingerprint"],
    })


@dg.asset(deps=[silver_tables], group_name="freshretailnet", compute_kind="scipy",
          description="Per-product demand parameters, methods 1/2/3, contract C1 (frn.gold).")
def demand_params(storage: FrnStorage, config: GoldConfig) -> dg.MaterializeResult:
    gold.build(storage.data_root, storage.source(), censor_threshold=config.censor_threshold,
               workers=config.workers)
    m = storage.manifest("gold")
    return dg.MaterializeResult(metadata={
        "products": m["summary"]["products"], "skipped_products": len(m["skipped_products"]),
        "censor_threshold": m["params"]["censor_threshold"], "summary": dg.MetadataValue.json(m["summary"]),
    })


# ---------------------------------------------------------------- checks (read what each layer recorded)

@dg.asset_check(asset=bronze_tables, description="Every error-level bronze check passed; warnings are reported.")
def bronze_checks_passed(storage: FrnStorage) -> dg.AssetCheckResult:
    checks = storage.manifest("bronze")["checks"]
    failed = [f"{s}: {c['name']}" for s, cs in checks.items() for c in cs if not c["passed"] and
              c.get("severity", "error") == "error"]
    warned = {f"{s}: {c['name']}": c["value"] for s, cs in checks.items() for c in cs if not c["passed"] and
              c.get("severity") == "warn"}
    return dg.AssetCheckResult(passed=not failed,
                               metadata={"failed": failed, "warnings": dg.MetadataValue.json(warned)})


@dg.asset_check(asset=silver_tables, description="All dbt data tests passed in the published build.")
def silver_dbt_tests_passed(storage: FrnStorage) -> dg.AssetCheckResult:
    t = storage.manifest("silver")["tests"]
    return dg.AssetCheckResult(passed=t["total"] > 0 and t["passed"] == t["total"], metadata=t)


@dg.asset_check(asset=demand_params, description="The published demand_params.parquet satisfies contract C1.")
def demand_params_contract(storage: FrnStorage) -> dg.AssetCheckResult:
    import pyarrow.parquet as pq

    problems = gold.validate(pq.read_table(storage.layer_dir("gold") / "demand_params.parquet"))
    return dg.AssetCheckResult(passed=not problems, metadata={"problems": problems})


@dg.asset_check(asset=demand_params, description="Sanity, not a contract: censoring-aware estimates should not be "
                "below the drop-stockout-days estimate for most products (warns only).")
def censoring_direction(storage: FrnStorage) -> dg.AssetCheckResult:
    s = storage.manifest("gold")["summary"]
    ok = s["median_lambda_ratio_m3_over_m1"] >= 1.0
    return dg.AssetCheckResult(passed=ok, severity=dg.AssetCheckSeverity.WARN,
                               metadata={"median_lambda_ratio_m3_over_m1": s["median_lambda_ratio_m3_over_m1"]})


pipeline = dg.define_asset_job("freshretailnet_pipeline", selection=dg.AssetSelection.groups("freshretailnet"))

defs = dg.Definitions(
    assets=[raw, bronze_tables, silver_tables, demand_params],
    asset_checks=[bronze_checks_passed, silver_dbt_tests_passed, demand_params_contract, censoring_direction],
    jobs=[pipeline],
    resources={"storage": FrnStorage()},
)
