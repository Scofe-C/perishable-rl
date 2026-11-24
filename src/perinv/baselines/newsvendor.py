"""Myopic newsvendor baseline (team draft §5: a simple reference that only looks one order ahead).

Today's order arrives on day t + τ, so it is the last chance to cover demand from today through that day. Per SKU,
order up to the critical-fractile quantile of Poisson demand over days t … t+τ (season-aware), on the inventory
position (on hand + in transit). Age and future expiry are ignored on purpose: that is what makes it myopic.

    cu = p − c + b    (a unit short: lost margin plus the lost-sales penalty)
    co = c + h + w    (a unit too many: its cost, a day of holding, and the expiry penalty if it goes to waste)
    S_t = PoissonQuantile(cu / (cu + co), Σ_{k=t}^{t+τ} λ·season[k mod 7])

Orders are then scaled to the remaining capacity. Deterministic; no tuning.

Owner: Member 2. Contract: docs/SPEC.md §8, WAREHOUSE.md §7.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from perinv.baselines.base import fit_to_capacity, inventory_position, remaining_capacity
from perinv.env.config import EnvConfig


def critical_ratio(cfg: EnvConfig) -> np.ndarray:
    p, c, h, w, b = (np.asarray(getattr(cfg, f), float) for f in
                     ("price", "order_cost", "holding_cost", "expiry_cost", "lost_sales_cost"))
    cu, co = p - c + b, c + h + w
    return np.clip(cu / (cu + co), 1e-6, 1 - 1e-6)


class NewsvendorPolicy:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        lam = np.asarray(cfg.demand_mean, float)
        season = np.asarray(cfg.season, float)
        cr = critical_ratio(cfg)
        # S[t, i] for every day: demand over the τ + 1 days from t (past the horizon the season pattern continues)
        days = np.arange(cfg.horizon)[:, None] + np.arange(cfg.lead_time + 1)[None, :]
        cover = season[days % 7].sum(axis=1)[:, None] * lam[None, :]              # [T, N]
        self.S = poisson.ppf(cr[None, :], cover).astype(np.int64)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        t = int(info["state"]["t"])
        q = np.maximum(self.S[t] - inventory_position(info), 0)
        return fit_to_capacity(q, remaining_capacity(info, self.cfg.capacity))


def build(ctx) -> NewsvendorPolicy:
    """Runner builder: built from ctx.cfg only (the policy's belief in a cross-evaluation)."""
    pol = NewsvendorPolicy(ctx.cfg)
    ctx.run.update(newsvendor={"critical_ratio": critical_ratio(ctx.cfg).tolist()})
    return pol
