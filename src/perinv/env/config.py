"""Environment parameters (docs/SPEC.md §1). Loaded from configs/env/*.yaml."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from perinv.data.scenarios import Shock, check_fits, parse_scenario


@dataclass(frozen=True)
class EnvConfig:
    n_skus: int
    capacity: int
    shelf_life: int
    lead_time: int
    horizon: int
    demand_mean: tuple[float, ...]
    season: tuple[float, ...] = (1.0,) * 7
    price: tuple[float, ...] = ()
    order_cost: tuple[float, ...] = ()
    holding_cost: tuple[float, ...] = ()
    expiry_cost: tuple[float, ...] = ()
    lost_sales_cost: tuple[float, ...] = ()
    kappa: float | None = None  # informational; capacity is what the env uses
    scenario: tuple[Shock, ...] = ()  # demand shocks on the CRN paths, eval only (perinv.data.scenarios)
    name: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.lead_time < 1:
            raise ValueError("lead_time must be >= 1 (pipeline has P = lead_time slots)")
        if len(self.season) != 7:
            raise ValueError("season needs 7 day-of-week multipliers")
        for f in ("demand_mean", "price", "order_cost", "holding_cost", "expiry_cost",
                  "lost_sales_cost"):
            if len(getattr(self, f)) != self.n_skus:
                raise ValueError(f"{f} must have one entry per SKU ({self.n_skus})")
        check_fits(self.scenario, self.n_skus, self.horizon)

    @property
    def pipeline_slots(self) -> int:
        return self.lead_time

    def daily_lambda(self) -> np.ndarray:
        """float64[T, N]: Poisson mean of SKU i on day t, lambda_i * season[t mod 7] (SPEC §3 step 3).
        The one definition shared by the env (D5), the CRN paths and the DP weights."""
        return (np.asarray(self.demand_mean, np.float64)[None, :]
                * np.asarray(self.season, np.float64)[np.arange(self.horizon) % 7, None])

    def demand_params(self) -> dict:
        """The subset that determines CRN demand paths (SPEC §7). `scenario` only when set (stable base hashes)."""
        d = {"N": self.n_skus, "lambda": list(self.demand_mean), "season": list(self.season), "T": self.horizon}
        if self.scenario:
            d["scenario"] = [s.to_dict() for s in self.scenario]
        return d

    def demand_hash(self) -> str:
        return stable_hash(self.demand_params())

    def config_hash(self) -> str:
        d = asdict(self)
        d.pop("name")
        if not self.scenario:
            d.pop("scenario")   # configs without a scenario keep the hash they had before the field existed
        return stable_hash(d)


def stable_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def capacity_from_kappa(kappa: float, ref_occupancy: float) -> int:
    """C = round(kappa * O_ref), SPEC §1.1."""
    return max(1, round(kappa * ref_occupancy))


# Top-level YAML keys that are not EnvConfig fields but are allowed as provenance (the loader ignores them).
PROVENANCE_KEYS = {"calibration", "shock_of"}
# Free extension namespace: any top-level key starting with this is ignored (notes, owner tags, tool metadata),
# so new annotations need no code change. They never affect the env or any hash.
EXTENSION_PREFIX = "x-"


def load_env_config(path: str | Path) -> EnvConfig:
    raw = yaml.safe_load(Path(path).read_text())
    known = (set(EnvConfig.__dataclass_fields__) - {"name"}) | PROVENANCE_KEYS
    unknown = {k for k in raw if k not in known and not str(k).startswith(EXTENSION_PREFIX)}
    if unknown:   # e.g. a misspelt `scenarios:` would otherwise load as the base world, silently
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; put free annotations under an "
                         f"'{EXTENSION_PREFIX}' prefix (e.g. x-note)")
    n = raw["n_skus"]

    def per_sku(key: str) -> tuple[float, ...]:
        v = raw[key]
        return tuple(float(x) for x in v) if isinstance(v, list) else (float(v),) * n

    return EnvConfig(
        n_skus=n,
        capacity=int(raw["capacity"]),
        shelf_life=int(raw["shelf_life"]),
        lead_time=int(raw["lead_time"]),
        horizon=int(raw["horizon"]),
        demand_mean=per_sku("demand_mean"),
        season=tuple(float(x) for x in raw.get("season", [1.0] * 7)),
        price=per_sku("price"),
        order_cost=per_sku("order_cost"),
        holding_cost=per_sku("holding_cost"),
        expiry_cost=per_sku("expiry_cost"),
        lost_sales_cost=per_sku("lost_sales_cost"),
        kappa=raw.get("kappa"),
        scenario=parse_scenario(raw.get("scenario")),
        name=Path(path).stem,
    )
