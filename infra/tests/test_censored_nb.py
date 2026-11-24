"""Simulation-based recovery: generate demand with known parameters, censor it like a store would, refit.

This is the main correctness argument for the gold layer: if the estimator cannot recover a truth we
planted, its numbers on real data mean nothing.
"""

import numpy as np
import pytest
from scipy import stats

from frn import censored_nb as cnb

TRUE_LAM, TRUE_R = 8.0, 3.0
TRUE_DOW = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.3, 1.0])
TRUE_DOW = TRUE_DOW / TRUE_DOW.mean()


def simulate(n_days=6000, seed=0, stock_mean=9.0):
    """Stock level is random and independent of demand; sales = min(demand, stock); censored = sold out."""
    rng = np.random.default_rng(seed)
    dow = rng.integers(0, 7, n_days)
    mu = TRUE_LAM * TRUE_DOW[dow]
    demand = rng.negative_binomial(TRUE_R, TRUE_R / (TRUE_R + mu))
    stock = rng.poisson(stock_mean, n_days)
    sales = np.minimum(demand, stock)
    cens = demand >= stock          # sold out: true demand is at least the sales
    return sales, dow, cens, demand


def test_logpmf_and_survival_match_scipy():
    y = np.array([0.0, 1, 3, 10, 25])
    mu, r = np.full(5, 4.0), 2.5
    ref = stats.nbinom(r, r / (r + 4.0))
    np.testing.assert_allclose(cnb._logpmf(y, r, mu), ref.logpmf(y), rtol=1e-10)
    np.testing.assert_allclose(cnb._logsf_ge(y, r, mu), np.log(np.maximum(ref.sf(y - 1), 1e-300)), rtol=1e-8)
    assert cnb._logsf_ge(np.array([0.0]), r, np.array([4.0]))[0] == 0.0     # P(D >= 0) = 1


def test_uncensored_fit_recovers_the_truth():
    _, dow, _, demand = simulate()
    f = cnb.fit(demand, dow, np.zeros_like(demand, dtype=bool))
    assert f.converged
    assert f.lam == pytest.approx(TRUE_LAM, rel=0.03)
    np.testing.assert_allclose(f.dow, TRUE_DOW, rtol=0.06)
    assert f.r == pytest.approx(TRUE_R, rel=0.15)
    assert f.dow.mean() == pytest.approx(1.0, abs=1e-12)


def test_censored_mle_removes_the_bias_that_dropping_stockout_days_keeps():
    sales, dow, cens, _ = simulate()
    assert 0.25 < cens.mean() < 0.6                                       # heavy censoring, like the real data
    m3 = cnb.fit(sales, dow, cens)                                        # method 3: all days, censored flagged
    m1 = cnb.fit(sales[~cens], dow[~cens], np.zeros((~cens).sum(), bool)) # method 1: drop stockout days
    naive = cnb.fit(sales, dow, np.zeros_like(cens))                      # treat sales as demand
    assert m3.lam == pytest.approx(TRUE_LAM, rel=0.04)
    assert m1.lam < 0.85 * TRUE_LAM and naive.lam < 0.85 * TRUE_LAM      # both biased low
    np.testing.assert_allclose(m3.dow, TRUE_DOW, rtol=0.08)


def test_censored_model_wins_out_of_sample():
    sales, dow, cens, _ = simulate(seed=1)
    test_sales, test_dow, test_cens, _ = simulate(seed=2, n_days=3000)
    m3 = cnb.fit(sales, dow, cens)
    m1 = cnb.fit(sales[~cens], dow[~cens], np.zeros((~cens).sum(), bool))
    assert cnb.heldout_loglik(m3, test_sales, test_dow, test_cens) > cnb.heldout_loglik(m1, test_sales, test_dow,
                                                                                         test_cens)


def test_weights_equal_repeated_rows():
    """Aggregating identical days into one weighted row must not change the likelihood (checked at fixed theta:
    with 4 points and 9 parameters the optimum is too flat to compare fitted values)."""
    y, dow, cens = np.array([2.0, 5.0, 5.0, 7.0]), np.array([0, 1, 1, 2]), np.array([False, True, True, False])
    theta = np.r_[np.log(4.0), np.linspace(-0.3, 0.3, 7), np.log(2.0)]
    a = cnb.loglik(theta, y, dow, cens, np.ones(4))
    b = cnb.loglik(theta, np.array([2.0, 5.0, 7.0]), np.array([0, 1, 2]), np.array([False, True, False]),
                   np.array([1.0, 2.0, 1.0]))
    assert a == pytest.approx(b, rel=1e-12)


def test_unseen_weekdays_get_factor_one():
    rng = np.random.default_rng(3)
    dow = rng.integers(0, 5, 2000)                                        # no weekend data at all
    y = rng.poisson(5.0, 2000).astype(float)
    f = cnb.fit(y, dow, np.zeros(2000, bool))
    assert f.dow.mean() == pytest.approx(1.0) and np.all(np.isfinite(f.dow))
