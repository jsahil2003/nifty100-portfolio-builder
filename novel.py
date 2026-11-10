"""
novel.py — our own ideas, fixed BEFORE they were run, to try to beat 1/N on the Nifty 100:

  1. Sector rotation (H4). Every rebalance, rank the 11 sectors by the average 6-month return of
     their eligible stocks. Only the top 5 sectors get money, by rank 5:4:3:2:1
     (33%, 27%, 20%, 13%, 7%). Inside each sector, the 3 best stocks by the MCDM score get equal
     shares (all of them if the sector has fewer than 3). Idea: sector momentum (Moskowitz &
     Grinblatt 1999) decides WHERE the money goes, the stock signals decide WHICH stocks.

  2. GARCH volatility management (H3, after Moreira & Muir 2017). Each rebalance, forecast the
     portfolio's own volatility with GARCH(1,1), and invest
        exposure = min(1, normal vol / forecast vol)
     in the stocks, the rest in cash at 6.5% (moving money to or from cash is free; the stock
     trades pay 0.1%). "Normal vol" = the portfolio's average volatility up to that day, so
     there is no setting to tune and no look-ahead.

Rule for "beats 1/N" (also fixed in advance): higher Sharpe than 1/N in BOTH the training
period and the test period. Each idea's alpha and beta against 1/N (backtest.py) show what it
added after adjusting for risk.
"""

import numpy as np
import pandas as pd

import allocation
import backtest as bt
import garch

TOP_SECTORS, PER_SECTOR, SECTOR_LOOKBACK = 5, 3, 126          # 126 trading days = 6 months
SECTOR_SHARES = np.arange(TOP_SECTORS, 0, -1) / np.arange(TOP_SECTORS, 0, -1).sum()   # 5/15 … 1/15


def sector_ranking(six_month_return, sectors, stocks):
    """Average 6-month return of the given stocks in each sector, best sector first."""
    return six_month_return[stocks].groupby(sectors[stocks]).mean().sort_values(ascending=False)


def sector_rotation_weights(score, six_month_return, sectors):
    """
    Weights of one rebalance. score: MCDM score of every eligible stock (higher = better);
    six_month_return: every stock's return over the past 6 months (same date).
    """
    ranking = sector_ranking(six_month_return, sectors, score.index)
    weights = {}
    for share, sector in zip(SECTOR_SHARES, ranking.index[:TOP_SECTORS]):
        best = score[sectors[score.index] == sector].nlargest(PER_SECTOR).index
        for stock in best:
            weights[stock] = share / len(best)
    return allocation.cap(pd.Series(weights))           # a 1-stock sector could exceed the 25% cap


def volatility_managed(stock_weights, strategy_returns, returns, label):
    """
    The strategy held with exposure e = min(1, normal vol / GARCH forecast vol), set at each
    of its own rebalance dates: the stocks get e × their usual weights, the rest (1 − e) is cash.
    The forecast uses only the unmanaged strategy's past daily returns (the last 2 years).
    Returns (daily returns, weights including CASH, exposure at each rebalance).
    """
    exposure = {}
    for date in stock_weights.index:
        past = strategy_returns.loc[:date].dropna().rename(label)     # the name keeps GARCH fits apart
        if len(past) < 252:                                  # not enough history yet: fully invested
            exposure[date] = 1.0
            continue
        forecast = garch.garch_volatility(past.tail(504))
        normal = past.std() * np.sqrt(252)
        exposure[date] = min(1.0, normal / forecast)
    exposure = pd.Series(exposure)
    weights = stock_weights.mul(exposure, axis=0)
    weights["CASH"] = 1 - exposure
    assets = returns.reindex(columns=stock_weights.columns).assign(CASH=bt.RISK_FREE / 252)
    return bt.portfolio_returns(weights, assets), weights, exposure
