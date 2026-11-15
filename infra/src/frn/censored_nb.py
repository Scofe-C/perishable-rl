"""Censored negative-binomial demand model with day-of-week effects. Pure math, no I/O.

Model for one product (all stores pooled):
    D ~ NegBin(mean = lam * f[dow], dispersion r)        f has mean 1 over the 7 days
    fully observed day  (sales = demand):   contributes  log P(D = y)
    censored day        (stockout, demand >= sales):  contributes  log P(D >= y)   (= 0 when y = 0)

The same likelihood serves all three estimation methods. They differ only in which rows go in and how they
are flagged. Rows are aggregated as (y, dow, censored, weight), so identical days cost one evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

LOG_R_BOUNDS = (-6.0, 9.0)  # r in [0.0025, 8100]; large r = nearly Poisson


@dataclass(frozen=True)
class Fit:
    lam: float                 # mean daily demand (count units), averaged over days of the week
    dow: np.ndarray            # 7 factors, mean exactly 1
    r: float                   # NB dispersion: Var = mu + mu^2 / r
    loglik: float              # weighted log-likelihood at the optimum
    converged: bool
    n_obs: int                 # weighted number of days used
    n_censored: int


def _logpmf(y, r, mu):
    """log NB pmf, parameterized by mean mu and dispersion r."""
    return (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def _logsf_ge(y, r, mu):
    """log P(D >= y) = log I_{mu/(r+mu)}(y, r) (regularized incomplete beta); 0 for y = 0."""
    p_success = mu / (r + mu)
    out = np.zeros_like(mu, dtype=float)
    pos = y > 0
    out[pos] = np.log(np.maximum(special.betainc(y[pos], r if np.isscalar(r) else r[pos], p_success[pos]), 1e-300))
    return out


def loglik(theta: np.ndarray, y: np.ndarray, dow: np.ndarray, cens: np.ndarray, w: np.ndarray) -> float:
    lam, f, r = unpack(theta)
    mu = lam * f[dow]
    ll = np.where(cens, _logsf_ge(y, r, mu), _logpmf(y, r, mu))
    return float(np.dot(w, ll))


def unpack(theta: np.ndarray) -> tuple[float, np.ndarray, float]:
    """theta = [log lam, g_0..g_6, log r]; f = exp(g) rescaled to mean 1 (lam absorbs the scale)."""
    e = np.exp(theta[1:8] - theta[1:8].max())
    f = e / e.mean()
    return float(np.exp(theta[0])), f, float(np.exp(theta[8]))


def fit(y, dow, cens, w=None, *, init_r: float = 1.0) -> Fit:
    """Maximum likelihood with L-BFGS-B (numeric gradient on aggregated rows; 9 parameters)."""
    y = np.asarray(y, dtype=float)
    dow = np.asarray(dow, dtype=int)
    cens = np.asarray(cens, dtype=bool)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if w.sum() == 0:
        raise ValueError("no observations")
    mean_y = max(float(np.dot(w, y) / w.sum()), 1e-3)
    theta0 = np.r_[np.log(mean_y), np.zeros(7), np.log(init_r)]
    bounds = [(np.log(1e-4), np.log(1e6))] + [(-5.0, 5.0)] * 7 + [LOG_R_BOUNDS]
    res = optimize.minimize(lambda th: -loglik(th, y, dow, cens, w), theta0, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 500})
    lam, f, r = unpack(res.x)
    # Days of the week that never appear carry no information; report factor 1 for them and renormalize.
    seen = np.bincount(dow, weights=w, minlength=7) > 0
    if not seen.all():
        f = np.where(seen, f, 1.0)
        f = f / f.mean()
    return Fit(lam=lam, dow=f, r=r, loglik=-float(res.fun), converged=bool(res.success),
               n_obs=round(w.sum()), n_censored=round(w[cens].sum()))


def heldout_loglik(model: Fit, y, dow, cens, w=None) -> float:
    """Mean censored log-likelihood per day of new data under a fitted model (higher is better)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    mu = model.lam * model.dow[np.asarray(dow, dtype=int)]
    cens = np.asarray(cens, dtype=bool)
    ll = np.where(cens, _logsf_ge(y, model.r, mu), _logpmf(y, model.r, mu))
    return float(np.dot(w, ll) / w.sum())
