"""
signals.py — the candidate stock signals: seven from published papers and one of our own.

Every signal is computed for every stock on every day at once with rolling windows (no loops),
using only past data. "Direction" says whether a HIGHER (+1) or a LOWER (-1) value is expected
to predict higher future returns. Only the signals that pass the training-period filter
(ranking.choose_signals) are used to pick stocks.
"""

import numpy as np

YEAR, HALF_YEAR, QUARTER, MONTH = 252, 126, 63, 21          # trading days

# name: (direction, source)
SIGNALS = {
    "momentum":      (+1, "Jegadeesh & Titman (1993): past 12-month winners keep winning"),
    "high_52w":      (+1, "George & Hwang (2004): stocks near their 52-week high do better"),
    "volatility":    (-1, "Ang et al. (2006), Blitz & van Vliet (2007): low-risk stocks do better per unit of risk"),
    "beta":          (-1, "Frazzini & Pedersen (2014): betting against beta"),
    "max_return":    (-1, "Bali, Cakici & Whitelaw (2011): 'lottery' stocks with huge up-days do worse"),
    "reversal":      (-1, "Jegadeesh (1990): last month's losers bounce back"),
    "illiquidity":   (+1, "Amihud (2002): less liquid stocks earn a premium"),
    "open_vs_med21": (-1, "our hypothesis: an open above its 21-day median open is a sell signal"),
}
DIRECTIONS = {name: direction for name, (direction, _) in SIGNALS.items()}
LABELS = {"momentum": "Momentum", "high_52w": "52-week high", "volatility": "Low volatility",
          "beta": "Low beta", "max_return": "Low MAX", "reversal": "Reversal",
          "illiquidity": "Illiquidity", "open_vs_med21": "Open vs 21-day median"}


def compute_signals(data):
    """{signal name: table (days × stocks)} for the eight candidate signals."""
    price, r, market = data["price"], data["returns"], data["market"]

    # beta = cov(stock, market) / var(market) over the past year, from rolling averages:
    # cov(x, y) = mean(x·y) − mean(x)·mean(y)
    mean_xy = r.mul(market, axis=0).rolling(YEAR).mean()
    cov = mean_xy - r.rolling(YEAR).mean().mul(market.rolling(YEAR).mean(), axis=0)
    beta = cov.div(market.rolling(YEAR).var(ddof=0), axis=0)

    rupees_traded = (data["close"] * data["volume"]).replace(0, np.nan)
    return {
        "momentum": price.shift(MONTH) / price.shift(YEAR) - 1,     # return from 12 months to 1 month ago
        "high_52w": price / price.rolling(YEAR).max(),             # 1.0 = at the 52-week high
        "volatility": r.rolling(YEAR).std() * np.sqrt(YEAR),       # annualised, past year
        "beta": beta,
        "max_return": r.rolling(MONTH).max(),                      # best single day of the past month
        "reversal": price / price.shift(MONTH) - 1,                # last month's return
        # price move per rupee traded; days with no volume data are skipped,
        # as long as at least half of the 3 months is available
        "illiquidity": np.log((r.abs() / rupees_traded).rolling(QUARTER, min_periods=QUARTER // 2).mean()),
        # today's open compared with the MEDIAN open of the last 21 days (including today):
        # +3% means the stock opened 3% above its typical opening price of the past month.
        # The median ignores one or two extreme opens (e.g. a gap after results), unlike the average.
        "open_vs_med21": data["open"] / data["open"].rolling(MONTH).median() - 1,
    }
