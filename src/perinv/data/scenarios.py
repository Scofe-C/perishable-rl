"""Demand-shock scenarios for stress-testing policies (evaluation only; docs/DECISIONS.md 2026-11-22).

A scenario is an optional list of shocks on an EnvConfig. It changes only the CRN demand paths; the env's own
dynamics and training demand are untouched, so a scenario config is only ever used as an `eval_envs` entry
(the runner enforces this). Train on the base world, score on the shocked one.

Coupling: path k of a shocked config is base path k (same draw, same rng) with the shock applied on top, using
a separate rng stream. So base-vs-shock differences per path isolate the shock instead of resampling noise.

Every shock is a per-day, per-SKU multiplier m on expected demand, applied to the realized base demand d:
    m > 1: d + Poisson(lambda * (m - 1))      m < 1: Binomial(d, m)
Both keep the marginal exactly Poisson(m * lambda) (Poisson superposition / thinning).

| kind          | magnitude                                    | example                         |
|---------------|----------------------------------------------|---------------------------------|
| spike         | multiplier > 1 (promotion, holiday)          | 3.0 = triple demand             |
| drop          | multiplier in [0, 1) (competitor, bad press) | 0.4 = 60 % of demand gone       |
| outage        | must be 0 (store closed): demand = 0         |                                 |
| overdispersed | phi > 0; m ~ Gamma(mean 1, var phi) per day   | var = lambda + phi * lambda^2   |

A permanent level shift is a spike or drop whose duration runs to the horizon. Shocks compose by multiplying.
The start day is drawn per path, uniformly in `start` = [lo, hi], so a policy cannot memorize the day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

KINDS = ("spike", "drop", "outage", "overdispersed")
MAX_SPIKE = 50.0          # keeps int32 demand and the env's state bounded
SHOCK_STREAM = 1          # rng([root_seed, split_id, k, SHOCK_STREAM]); base paths use the 3-element seed
_FIELDS = {"kind", "start", "duration", "magnitude", "skus"}


@dataclass(frozen=True)
class Shock:
    kind: str
    start: tuple[int, int]          # start day drawn uniformly in [lo, hi] (inclusive), per path
    duration: int                   # days; truncated at the horizon
    magnitude: float = 0.0
    skus: tuple[int, ...] | None = None   # None = every SKU

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"shock kind {self.kind!r} is not one of {KINDS}")
        lo, hi = self.start
        if not 0 <= lo <= hi:
            raise ValueError(f"shock start window {self.start} must satisfy 0 <= lo <= hi")
        if self.duration < 1:
            raise ValueError(f"shock duration must be >= 1, got {self.duration}")
        m = self.magnitude
        ok = {"spike": 1.0 < m <= MAX_SPIKE, "drop": 0.0 <= m < 1.0, "outage": m == 0.0,
              "overdispersed": m > 0.0}[self.kind]
        if not ok:
            raise ValueError(f"{self.kind} shock: magnitude {m} out of range (see perinv.data.scenarios)")
        if self.skus is not None and (not self.skus or min(self.skus) < 0 or len(set(self.skus)) != len(self.skus)):
            raise ValueError(f"shock skus {self.skus} must be distinct non-negative indices")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = list(self.start)
        d["skus"] = list(self.skus) if self.skus is not None else None
        return d


def parse_scenario(raw: list | None) -> tuple[Shock, ...]:
    """YAML list -> shocks. Unknown keys are an error: a typo must never silently fall back to the base world."""
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ValueError("scenario must be a non-empty list of shocks")
    out = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise ValueError(f"scenario[{i}] must be a mapping, got {s!r}")
        unknown = set(s) - _FIELDS
        if unknown:
            raise ValueError(f"scenario[{i}]: unknown keys {sorted(unknown)}; allowed {sorted(_FIELDS)}")
        start = s["start"]
        start = (int(start), int(start)) if isinstance(start, int) else tuple(int(x) for x in start)
        if len(start) != 2:
            raise ValueError(f"scenario[{i}]: start must be a day or [lo, hi]")
        skus = s.get("skus")
        out.append(Shock(kind=s["kind"], start=start, duration=int(s["duration"]),
                         magnitude=float(s.get("magnitude", 0.0)),
                         skus=tuple(int(x) for x in skus) if skus is not None else None))
    return tuple(out)


def check_fits(shocks: tuple[Shock, ...], n_skus: int, horizon: int) -> None:
    for s in shocks:
        if s.start[1] > horizon - 1:
            raise ValueError(f"{s.kind} shock may start on day {s.start[1]}, past the horizon {horizon}")
        if s.skus is not None and max(s.skus) >= n_skus:
            raise ValueError(f"{s.kind} shock targets SKU {max(s.skus)}, but the env has {n_skus}")


def multipliers(shocks: tuple[Shock, ...], horizon: int, n_skus: int, rng: np.random.Generator) -> np.ndarray:
    """float[T, N] demand multipliers for one path. Draw order is fixed: per shock, start day then gamma."""
    m = np.ones((horizon, n_skus))
    for s in shocks:
        t0 = int(rng.integers(s.start[0], s.start[1] + 1))
        days = slice(t0, min(t0 + s.duration, horizon))
        cols = list(s.skus) if s.skus is not None else slice(None)
        if s.kind == "overdispersed":
            n_days = days.stop - days.start
            n_cols = len(s.skus) if s.skus is not None else n_skus
            m[days, cols] *= rng.gamma(1.0 / s.magnitude, s.magnitude, size=(n_days, n_cols))
        else:
            m[days, cols] *= s.magnitude
    return m


def apply_shocks(base: np.ndarray, lam: np.ndarray, shocks: tuple[Shock, ...],
                 rng: np.random.Generator) -> np.ndarray:
    """base int[T, N] realized demand, lam float[T, N] its Poisson means -> shocked int32[T, N]."""
    m = multipliers(shocks, base.shape[0], base.shape[1], rng)
    extra = rng.poisson(lam * np.maximum(m - 1.0, 0.0))       # zero where m <= 1
    kept = rng.binomial(base, np.minimum(m, 1.0))             # = base where m >= 1
    return (kept + extra).astype(np.int32)
