"""
allocation.py — how much money each selected stock gets. Every function takes the past
year of daily returns of the selected stocks (days × stocks) and returns weights that sum
to 1, long-only, at most 25% per stock.

  1/N                   equal weights                       DeMiguel, Garlappi & Uppal (2009)
  Inverse volatility    weight ∝ 1 / volatility             (risk-based baseline)
  Minimum variance      lowest-risk mix                     Markowitz (1952); Ledoit & Wolf (2004)
  Max Sharpe            best return per unit of risk        Markowitz (1952)
  HRP                   Hierarchical Risk Parity            López de Prado (2016)
  Inverse GARCH vol     weight ∝ 1 / FORECAST volatility    Bollerslev (1986)   (garch.py)
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

import garch

STOCK_CAP = 0.25
RISK_FREE = 0.065            # Indian T-bill rate, per year


def covariance(window):
    """Annual covariance matrix, shrunk towards a simpler target (Ledoit-Wolf) to reduce noise."""
    values = LedoitWolf().fit(window.fillna(0).values).covariance_ * 252
    return pd.DataFrame(values, index=window.columns, columns=window.columns)


def cap(w):
    """Push weights under STOCK_CAP: clip, and share the excess among the stocks below the cap."""
    w = w / w.sum()
    for _ in range(50):
        over = w > STOCK_CAP
        if not over.any():
            break
        excess = (w[over] - STOCK_CAP).sum()
        w[over] = STOCK_CAP
        w[~over] += excess * w[~over] / w[~over].sum()
    return w


def equal_weight(window):
    return pd.Series(1 / window.shape[1], index=window.columns)


def inverse_volatility(window):
    return cap(1 / window.std())


def inverse_garch_volatility(window):
    """Like inverse volatility, but with each stock's GARCH(1,1) forecast of next month's volatility."""
    forecast = pd.Series({stock: garch.garch_volatility(window[stock]) for stock in window.columns})
    return cap(1 / forecast)


def minimum_variance(window):
    """min wᵀΣw  s.t.  Σw = 1, 0 ≤ w ≤ 25%."""
    sigma = covariance(window).values
    w = cp.Variable(len(sigma))
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), [cp.sum(w) == 1, w >= 0, w <= STOCK_CAP]).solve()
    return pd.Series(np.clip(w.value, 0, None), index=window.columns)


def max_sharpe(window):
    """
    max (μ − r_f)ᵀw / √(wᵀΣw), with μ = average past return × 252. Substituting
    y = w / ((μ − r_f)ᵀw) makes it the easy problem  min yᵀΣy  s.t. (μ − r_f)ᵀy = 1, y ≥ 0,
    y ≤ 25%·Σy;  then w = y / Σy. If too few stocks beat the risk-free rate (the problem has no
    solution under the caps), use minimum variance.
    """
    excess = window.mean().values * 252 - RISK_FREE
    if excess.max() <= 0:
        return minimum_variance(window)
    sigma = covariance(window).values
    y = cp.Variable(len(sigma))
    constraints = [excess @ y == 1, y >= 0, y <= STOCK_CAP * cp.sum(y)]
    cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))), constraints).solve()
    if y.value is None:                     # too few stocks beat r_f to fill the 25% caps
        return minimum_variance(window)
    y = np.clip(y.value, 0, None)
    return pd.Series(y / y.sum(), index=window.columns)


def cluster_variance(cov, members):
    """Variance of a cluster held with inverse-variance weights."""
    c = cov.loc[members, members].values
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return ivp @ c @ ivp


def hrp(window):
    """
    Hierarchical Risk Parity (López de Prado 2016):
      1. cluster stocks by correlation (distance √((1 − ρ)/2)), order similar stocks together;
      2. split the ordered list in halves again and again, giving each half money in inverse
         proportion to its risk.
    No matrix inversion, so it is robust to noisy estimates.
    """
    cov = covariance(window)
    sd = np.sqrt(np.diag(cov))
    distance = np.sqrt(np.clip((1 - cov / np.outer(sd, sd)) / 2, 0, None))
    order = cov.index[leaves_list(linkage(squareform(distance.values, checks=False), "single"))]
    w = pd.Series(1.0, index=order)
    clusters = [list(order)]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for left, right in zip(clusters[::2], clusters[1::2]):
            v_left, v_right = cluster_variance(cov, left), cluster_variance(cov, right)
            w[left] *= v_right / (v_left + v_right)          # the riskier half gets less
            w[right] *= v_left / (v_left + v_right)
    return cap(w.reindex(window.columns))


ALLOCATORS = {"1/N": equal_weight, "Inverse volatility": inverse_volatility,
              "Inverse GARCH vol": inverse_garch_volatility,
              "Minimum variance": minimum_variance, "Max Sharpe": max_sharpe, "HRP": hrp}
