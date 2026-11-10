"""
data.py — loads the project's data (daily, January 2015 – July 2026).

All files are in data_cache/pit/ ("point in time"), one Yahoo Finance download, so open, high,
low and close are adjusted for dividends and splits in the same way:
  open/high/low/close.csv   adjusted daily prices of the 172 stocks that were in the Nifty 100 at
                            some point since 2016, including stocks that later left it (Yes Bank, ...)
  volume.csv                number of shares traded each day
  members.csv               1 if the stock was in the Nifty 100 ON THAT DAY, else 0 (rebuilt from
                            NSE's official inclusion/exclusion records)
  sectors.csv               the sector of each stock (Yahoo's classification, 11 sectors)
  sensex.csv                the BSE Sensex index (used as "the market")

Using each day's members (not today's list) avoids survivorship bias: an investor in 2016 could
only buy the 2016 index, including stocks that later collapsed. On today's list the 1/N portfolio
earned 23.5% a year; on the true lists only 15.3%.
"""

import pandas as pd

FOLDER = "data_cache/pit"
TRAIN_END = "2022-12-31"                           # every choice is made on 2016 – 2022 ...
TEST_START, TEST_END = "2023-01-01", "2026-07-31"  # ... and checked on 43 later months

# Days on which Yahoo shows a fake price move (a demerger or an unadjusted bonus issue).
# Their return is set to 0. Every other move beyond ±17% was checked against the news.
CORPORATE_ACTIONS = [("ADANIENT", "2015-06-03"), ("MOTHERSON", "2022-01-14"), ("TMPV", "2025-10-14"),
                     ("TRENT", "2026-01-01"), ("VEDL", "2026-04-30"), ("CONCOR", "2025-01-01")]


def read(name):
    return pd.read_csv(f"{FOLDER}/{name}.csv", index_col=0, parse_dates=True)


def load():
    """All tables, indexed by trading day. Returns are simple daily returns (NaN = no price that day)."""
    close, volume = read("close"), read("volume")
    # Yahoo lists a few market holidays as rows with unchanged prices and zero volume (e.g. 1 May,
    # Maharashtra Day). Keep only days on which at least half of the stocks with a price traded.
    traded = (volume > 0).sum(axis=1) >= 0.5 * close.notna().sum(axis=1)
    tables = {name: read(name).loc[traded] for name in ["open", "high", "low", "close"]}
    returns = tables["close"].pct_change(fill_method=None)
    for ticker, day in CORPORATE_ACTIONS:
        if pd.Timestamp(day) in returns.index:
            returns.loc[day, ticker] = 0.0
    sensex = read("sensex").iloc[:, 0].reindex(returns.index).ffill()
    return {**tables, "volume": volume.loc[traded],
            "members": read("members").reindex(returns.index).fillna(0).astype(bool),
            "sectors": pd.read_csv(f"{FOLDER}/sectors.csv", index_col=0).iloc[:, 0],
            "returns": returns,
            "price": (1 + returns.fillna(0)).cumprod().where(returns.notna() | tables["close"].notna()),
            "market": sensex.pct_change(fill_method=None)}
