"""
garch.py — GARCH(1,1) volatility forecast (Bollerslev 1986).

Today's variance = ω + α·(yesterday's shock)² + β·(yesterday's variance): volatility comes in
clusters and slowly returns to its long-run level. Used to size positions (allocation.py), for
volatility management (novel.py) and by the web app (app.py).

It lives in its own file (it used to be in prediction.py) so that code which only needs GARCH
does not have to load PyTorch, which prediction.py needs for the LSTM.
"""

import warnings

import numpy as np
from arch import arch_model

_GARCH_CACHE = {}


def garch_volatility(returns):
    """
    Next-month volatility (annualised) of one stock from a GARCH(1,1) fitted to its past
    returns. Falls back to the plain historical volatility if the fit fails.
    """
    key = (returns.name, returns.index[-1])
    if key not in _GARCH_CACHE:
        r = returns.dropna() * 100                          # percent: the optimiser prefers numbers near 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant").fit(disp="off")
                variance_21d = fit.forecast(horizon=21).variance.values[-1].sum()   # next 21 days, in %²
            _GARCH_CACHE[key] = np.sqrt(variance_21d * 252 / 21) / 100
        except Exception:
            _GARCH_CACHE[key] = r.std() * np.sqrt(252) / 100
    return _GARCH_CACHE[key]
