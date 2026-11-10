"""
backtest.py — turn monthly target weights into daily portfolio returns, and measure them.

How the simulation works (₹1 crore, monthly):
  * At the close of the last trading day of each month, the strategy decides target weights
    using only data up to that day.
  * It trades at the NEXT day's close (you cannot trade at the price that produced the signal).
  * Between trades, the number of shares is fixed, so weights drift with prices (buy and hold).
  * Every trade costs 0.1% of the value traded (brokerage, taxes, spread).
Everything is computed with whole-table operations; the only grouping is by holding period.
"""

import numpy as np
import pandas as pd

CAPITAL = 10_000_000            # ₹1 crore
COST = 0.001                    # 0.1% of the value traded
RISK_FREE = 0.065               # per year, for the Sharpe ratio


def simulate(weights, returns):
    """
    weights: decision dates × assets (each row sums to 1; an asset called "CASH" earns the
    risk-free rate and is free to trade). Holding period k = the days after trade k up to
    and including trade k+1 (the trade day still earns the old portfolio's return; the new
    one starts the day after). Returns the daily net returns and the trade details.
    """
    days = returns.index
    trade_days = days[days.searchsorted(weights.index, side="right")]      # next trading day after each decision
    period = pd.Series(trade_days.searchsorted(days, side="left") - 1, index=days)   # last trade before each day
    period = period[period >= 0]                                           # not invested before the first trade
    r = returns.reindex(index=period.index, columns=weights.columns).fillna(0.0)   # no price = no change

    held = pd.DataFrame(weights.values[period.values], index=r.index, columns=weights.columns)
    growth = (1 + r).groupby(period.values).cumprod()        # each asset's growth since the last trade
    value = (held * growth).sum(axis=1)                      # value of ₹1 invested at the last trade
    gross = value / value.groupby(period.values).shift(1).fillna(1.0) - 1

    # costs: turnover = Σ |new weight − weight the old portfolio has drifted to| (cash is free)
    drifted = (held * growth).groupby(period.values).last().div(value.groupby(period.values).last(), axis=0)
    previous = np.vstack([np.zeros(weights.shape[1]), drifted.values[:-1]])      # first trade starts from cash
    stocks = weights.columns != "CASH"
    turnover = np.abs(weights.values[:len(drifted)] - previous)[:, stocks].sum(axis=1)
    cost_days = [period.index[0]] + list(trade_days[1:len(drifted)])             # where each trade's cost is paid
    cost = pd.Series(COST * turnover, index=cost_days).reindex(r.index).fillna(0.0)
    return {"net": (1 + gross) * (1 - cost) - 1, "trade_days": trade_days[:len(drifted)],
            "before": pd.DataFrame(previous, columns=weights.columns), "cost_rate": COST * turnover}


def portfolio_returns(weights, returns):
    """Daily net returns of a strategy (see simulate)."""
    return simulate(weights, returns)["net"]


def trade_log(weights, returns, close, name):
    """
    Every trade of one strategy, in rupees, starting from ₹1 crore: for each rebalance, the
    change from the drifted old weight to the new weight × the portfolio value at that moment.
    (The backtest trades exact rupee amounts; `shares` is that amount ÷ the closing price,
    rounded to a whole number.)
    """
    sim = simulate(weights, returns)
    wealth = (1 + sim["net"]).cumprod()
    prices = close.ffill()
    rows = []
    for k, day in enumerate(sim["trade_days"]):
        if k == 0:
            value = CAPITAL                                              # start: ₹1 crore in cash
        else:
            value = CAPITAL * wealth.loc[day] / (1 - sim["cost_rate"][k])  # value just before this trade's cost
        change = (weights.iloc[k].values - sim["before"].iloc[k].values) * value
        for ticker, rupees, before, after in zip(weights.columns, change, sim["before"].iloc[k], weights.iloc[k]):
            if ticker == "CASH" or abs(rupees) < 1:
                continue
            price = prices.at[day, ticker]
            rows.append({"strategy": name, "decision_date": weights.index[k].date(), "trade_date": day.date(),
                         "ticker": ticker, "side": "BUY" if rupees > 0 else "SELL",
                         "value_₹": round(abs(rupees), 2), "price_₹": round(price, 2),
                         "shares": int(round(abs(rupees) / price)), "cost_₹": round(COST * abs(rupees), 2),
                         "weight_before_%": round(before * 100, 3), "weight_after_%": round(after * 100, 3)})
    return pd.DataFrame(rows)


def metrics(daily, start=None, end=None):
    """Headline numbers of one strategy over [start, end]."""
    r = daily.loc[start:end]
    wealth = (1 + r).cumprod()
    years = len(r) / 252
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    return {"ann_return_%": (wealth.iloc[-1] ** (1 / years) - 1) * 100,
            "final_value_₹_crore": wealth.iloc[-1] * CAPITAL / 1e7,
            "pnl_₹_crore": (wealth.iloc[-1] - 1) * CAPITAL / 1e7,
            "ann_volatility_%": r.std() * np.sqrt(252) * 100,
            "sharpe": (r.mean() - RISK_FREE / 252) / r.std() * np.sqrt(252),
            "max_drawdown_%": (wealth / wealth.cummax() - 1).min() * 100,
            "positive_months_%": (monthly > 0).mean() * 100}


def sharpe(daily):
    return (daily.mean() - RISK_FREE / 252) / daily.std() * np.sqrt(252)


def monthly_excess(daily):
    """Monthly returns (compounded from the daily ones) minus the monthly risk-free rate."""
    monthly = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    return monthly - RISK_FREE / 12


def alpha_beta(a, b):
    """
    Jensen's alpha and beta of strategy a against the benchmark b, on MONTHLY excess returns:
        a_t − r_f = α + β (b_t − r_f) + ε_t        (straight line fitted by least squares)
    β = how much of the strategy is just "the benchmark with more or less risk"
        (β = 1.2: it moves 1.2 times as much as the benchmark);
    α = the average extra return per month that is left after that risk adjustment, i.e. what the
        stock picking itself added. Reported in % a year (× 12).
    """
    y, x = monthly_excess(a).align(monthly_excess(b), join="inner")
    beta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)        # slope of the fitted line
    alpha = y.mean() - beta * x.mean()                        # intercept of the fitted line
    return {"alpha_%_per_year": alpha * 12 * 100, "beta": beta}
