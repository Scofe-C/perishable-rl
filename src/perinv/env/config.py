"""Environment parameters (docs/SPEC.md §1). Loaded from configs/env/*.yaml."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


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

    @property
    def pipeline_slots(self) -> int:
        return self.lead_time

    def demand_params(self) -> dict:
        """The subset that determines CRN demand paths (SPEC §7)."""
        return {"N": self.n_skus, "lambda": list(self.demand_mean),
                "season": list(self.season), "T": self.horizon}

    def demand_hash(self) -> str:
        return stable_hash(self.demand_params())

    def config_hash(self) -> str:
        d = asdict(self)
        d.pop("name")
        return stable_hash(d)


def stable_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def capacity_from_kappa(kappa: float, ref_occupancy: float) -> int:
    """C = round(kappa * O_ref), SPEC §1.1."""
    return max(1, round(kappa * ref_occupancy))


def load_env_config(path: str | Path) -> EnvConfig:
    raw = yaml.safe_load(Path(path).read_text())
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
        name=Path(path).stem,
    )
