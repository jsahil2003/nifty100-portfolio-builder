"""
app.py — the portfolio builder page.  Run:  streamlit run app.py

  1. The user enters their capital.
  2. They choose a strategy and see which stocks to buy, how many whole shares, and how much money
     goes into stocks, into a liquid fund, and stays as cash:
       Strategy 1: 1/N               every Nifty 100 member, the same money in each (the study's 1/N)
       Strategy 2: MCDM + GARCH      the 15 MCDM picks (CRITIC + weighted sum), the same money in each,
                                     times the GARCH exposure e = min(1, normal vol / forecast vol); the
                                     rest (1 − e) goes to the liquid fund (the study's H3)
       Strategy 3: Sector rotation   the 5 sectors with the best 6-month return get money 5:4:3:2:1,
                                     split equally between their 3 best MCDM stocks (the study's H4)
  3. Expected return, volatility and drawdown = the AVERAGE over the backtest's full calendar years
     (2017–2025) of each year's return, annualised volatility and worst fall within the year.
     Partial years (Feb–Dec 2016, Jan–Jul 2026) are left out so that every year counts the same.

The page is written as a product: it does not describe the study. Everything is taken from the
study's own code and saved results (results/daily_returns.csv, results/main_scheme.txt), nothing is
re-estimated.
"""

import numpy as np
import pandas as pd
import streamlit as st

import backtest as bt
import data as dt
import novel
import garch
import ranking as rk
from signals import compute_signals

STOCKS = 15                                     # Strategy 2 holds the 15 best-ranked stocks
SCHEME = "CRITIC + weighted sum"                # the study's main MCDM scheme
EQUAL, MANAGED, ROTATION = "Strategy 1: 1/N", "Strategy 2: MCDM + GARCH", "Strategy 3: Sector rotation"
DESCRIPTION = {EQUAL: "Every Nifty 100 stock, the same amount in each.",
               MANAGED: "15 selected stocks, the same amount in each. When markets look riskier than "
                        "usual, part of the money waits in a liquid fund.",
               ROTATION: "The 5 strongest sectors, 3 selected stocks in each. Stronger sectors get more "
                         "money. Higher risk, higher potential return."}
HISTORY_COLUMN = {EQUAL: "1/N (all Nifty 100 members)",                    # results/daily_returns.csv
                  MANAGED: "CRITIC + weighted sum, 1/N, monthly + GARCH vol-managed",
                  ROTATION: "Sector rotation, monthly"}
UNMANAGED_COLUMN = "CRITIC + weighted sum, 1/N, monthly"   # its GARCH forecast drives the exposure
FULL_YEARS = ("2017", "2025")                   # the complete calendar years of the backtest


# ============================================================================ step 2a: the ranking
@st.cache_data
def latest_ranking(date=None):
    """
    Every Nifty 100 member on `date` (default: the last date in the data): sector, closing price (₹),
    6-month return, MCDM score and rank (1 = best). As in main.py, a stock is scored only if ALL 8
    signals exist (about a year of history); otherwise its score and rank are empty.
    """
    data = dt.load()
    signals = compute_signals(data)
    names = open("results/main_scheme.txt").read().splitlines()[2].split(", ")   # the 3 kept signals
    date = data["close"].index[-1] if date is None else pd.Timestamp(date)
    members = data["members"].loc[date] & data["close"].loc[date].notna()
    has_signals = pd.concat([table.loc[date].notna() for table in signals.values()], axis=1).all(axis=1)
    matrix = rk.decision_matrix(signals, date, list(members.index[members & has_signals]), names)
    price = data["price"]                                     # return index, corporate actions removed
    table = pd.DataFrame({"sector": data["sectors"].reindex(members.index[members]),
                          "price": data["close"].loc[date, members],
                          "six_month": (price.loc[date] / price.shift(novel.SECTOR_LOOKBACK).loc[date] - 1)[members]})
    table["score"] = rk.scores(SCHEME, matrix, None)          # CRITIC needs no BWM weights
    table["rank"] = table["score"].rank(ascending=False)
    return date, table.sort_values("rank")


@st.cache_data
def stock_exposure():
    """
    Share of the money Strategy 2 puts in stocks (as in novel.volatility_managed):
        e = min(1, normal vol / GARCH forecast vol)
    normal vol = the 15-stock portfolio's volatility over its whole history; the forecast is a
    GARCH(1,1) on its last 2 years of daily returns.
    """
    past = pd.read_csv("results/daily_returns.csv", index_col=0, parse_dates=True)[UNMANAGED_COLUMN].dropna()
    normal = past.std() * np.sqrt(252)
    return min(1.0, normal / garch.garch_volatility(past.tail(504)))


def target_weights(strategy, ranking):
    """
    Share of the capital for each stock (sums to 1, or to e < 1 for Strategy 2: the rest is the liquid fund).
      1: 1/N on every member;  2: e / 15 on the 15 best-ranked;  3: novel.sector_rotation_weights.
    """
    if strategy == EQUAL:
        return pd.Series(1 / len(ranking), index=ranking.index)
    if strategy == MANAGED:
        return pd.Series(stock_exposure() / STOCKS, index=ranking.index[:STOCKS])
    scored = ranking.dropna(subset=["score"])                 # only scored stocks can be picked
    return novel.sector_rotation_weights(scored["score"], scored["six_month"], scored["sector"])


# ============================================================================ step 2b: the shopping list
def buy_list(weights, ranking, capital):
    """
    Whole shares for each target weight:
      budget = capital × weight × (1 − 0.1% cost);  shares = floor(budget / price).
    A stock dearer than its budget gets 0 shares.
    """
    out = ranking.loc[weights.index, ["sector", "price"]]
    out["shares"] = np.floor(capital * weights * (1 - bt.COST) / out["price"]).astype(int)
    out["amount"] = out["shares"] * out["price"]
    return out


def inr(x):
    """₹ with Indian digit grouping: 1234567 → ₹12,34,567."""
    whole = f"{abs(x):.0f}"
    last3, rest = whole[-3:], whole[:-3]
    groups = [rest[max(i - 2, 0):i] for i in range(len(rest), 0, -2)][::-1]   # pairs from the right
    return ("-" if x < 0 else "") + "₹" + ",".join(groups + [last3])


# ============================================================================ step 3: expected numbers
@st.cache_data
def expected(strategy):
    """
    Average over the full backtest years of: that year's return, its annualised volatility
    (std of daily returns × √252) and its worst fall from a peak within the year. In %.
    """
    daily = pd.read_csv("results/daily_returns.csv", index_col=0, parse_dates=True)[HISTORY_COLUMN[strategy]]
    daily = daily.loc[FULL_YEARS[0]:FULL_YEARS[1]]
    by_year = daily.groupby(daily.index.year)
    returns = by_year.apply(lambda r: (1 + r).prod() - 1)
    volatility = by_year.std() * np.sqrt(252)
    drawdown = by_year.apply(lambda r: ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
    return returns.mean() * 100, volatility.mean() * 100, drawdown.mean() * 100


# ============================================================================ the page
st.set_page_config(page_title="Nifty 100 Portfolio Builder", page_icon="📈")
st.title("Nifty 100 Portfolio Builder")
st.info("For educational purposes only. This is not investment advice.")

# ---------------------------------------------------------------- 1. capital
st.header("1. Your capital")
capital = st.number_input("Amount to invest (₹)", min_value=10_000, value=5_00_000, step=10_000)

# ---------------------------------------------------------------- 2. strategy and what to buy
st.header("2. Your portfolio")
strategy = st.radio("Strategy", [EQUAL, MANAGED, ROTATION], captions=[DESCRIPTION[s] for s in (EQUAL, MANAGED, ROTATION)])
date, ranking = latest_ranking()
weights = target_weights(strategy, ranking)
basket = buy_list(weights, ranking, capital)
invested = basket["amount"].sum()
fund = capital * (1 - weights.sum())                           # liquid fund (only Strategy 2, in risky times)
cash = capital - fund - invested * (1 + bt.COST)               # what whole shares and the 0.1% cost leave

c1, c2, c3 = st.columns(3)
c1.metric("Invest in stocks", inr(invested))
c2.metric("Liquid fund", inr(fund))
c3.metric("Left as cash", inr(cash))
too_dear = basket.index[basket["shares"] == 0]
if len(too_dear):
    st.warning(f"Your capital is too small to buy one share of {len(too_dear)} of the {len(basket)} stocks: "
               f"{', '.join(too_dear)}.")
table = basket[basket["shares"] > 0].assign(weight=lambda t: t["amount"] / capital * 100)
table = table.rename(columns={"sector": "Sector", "price": "Price (₹)", "shares": "Shares",
                              "amount": "Amount (₹)", "weight": "% of capital"})
st.dataframe(table.style.format({"Price (₹)": "{:,.2f}", "Amount (₹)": "{:,.0f}", "% of capital": "{:.2f}"}))
st.caption(f"Prices as of {date:%d %b %Y}. Rebalance at the end of every month. "
           "Trading cost of 0.1% included.")

# ---------------------------------------------------------------- 3. expected numbers
st.header("3. What to expect in a year")
ret, vol, dd = expected(strategy)
c1, c2, c3 = st.columns(3)
c1.metric("Expected return", f"{ret:.1f}%", f"{inr(capital * ret / 100)} a year", delta_color="off")
c2.metric("Expected volatility", f"{vol:.1f}%", f"± {inr(capital * vol / 100)}", delta_color="off")
c3.metric("Expected drawdown", f"{dd:.1f}%", f"{inr(capital * dd / 100)} at the worst point", delta_color="off")
st.caption(f"Averages over the calendar years {FULL_YEARS[0]}–{FULL_YEARS[1]} of a historical simulation, "
           "after trading costs. Past performance does not guarantee future returns. "
           "For educational purposes only; not investment advice.")
