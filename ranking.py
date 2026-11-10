"""
ranking.py — choose the signals, then combine them into one score per stock (multi-criteria
decision making, MCDM).

  0. CHOOSE the signals on the training period only (choose_signals):
       keep a signal if its information ratio ICIR = mean IC / std IC is above 0.05, and of two
       signals that mostly say the same thing (correlation above 0.6), keep only the better one.
  1. WEIGHT the chosen signals:  Best-Worst Method (expert judgement, Rezaei 2015/2016)
                                 | CRITIC (from the data, Diakoulaki et al. 1995)
  2. RANK the stocks:            weighted sum | TOPSIS (Hwang & Yoon 1981)
Two schemes are compared: CRITIC + weighted sum and BWM + TOPSIS.

The decision matrix of one month is stocks × signals, each signal turned into a z-score with
HIGHER = BETTER (signals where lower is better are multiplied by −1).
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from signals import DIRECTIONS

MIN_ICIR = 0.05            # a signal must have at least this monthly information ratio on training data
MAX_CORRELATION = 0.6      # two signals more correlated than this are treated as one signal


# ============================================================================ step 0: choosing the signals
def information_coefficients(signal_tables, next_month_returns, eligible, names):
    """
    IC = Spearman rank correlation, across stocks, between a signal at month-end and the
    stocks' return over the next month (direction-adjusted, so positive = the signal worked).
    Computed for all months at once: rank each row, then a row-by-row correlation.
    Returns a table months × signals.
    """
    ics = {}
    for name in names:
        signal = signal_tables[name].loc[next_month_returns.index] * DIRECTIONS[name]
        both = eligible & signal.notna() & next_month_returns.notna()
        a = signal.where(both).rank(axis=1)
        b = next_month_returns.where(both).rank(axis=1)
        a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
        ics[name] = (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return pd.DataFrame(ics)


def signal_correlation(signal_tables, eligible, names):
    """
    How similar are two signals? Each month-end: the Spearman correlation between the signals
    across the eligible stocks (direction-adjusted). Averaged over the given months.
    """
    monthly = [pd.DataFrame({n: signal_tables[n].loc[date, eligible.loc[date]] * DIRECTIONS[n] for n in names})
               .corr(method="spearman") for date in eligible.index]
    return sum(monthly) / len(monthly)


def choose_signals(train_ic, correlation):
    """
    The signal filter (training data only). Returns (kept signals, table with the reason for each).
      1. ICIR = mean IC / std IC must exceed MIN_ICIR (the signal predicted returns reasonably often).
      2. Go through the survivors from best to worst ICIR; drop a signal if it is correlated
         above MAX_CORRELATION with a signal already kept (it adds little new information).
    """
    icir = train_ic.mean() / train_ic.std()
    kept, reason = [], {}
    for name in icir.sort_values(ascending=False).index:
        twin = [k for k in kept if abs(correlation.loc[name, k]) > MAX_CORRELATION]
        if icir[name] <= MIN_ICIR:
            reason[name] = f"dropped: ICIR {icir[name]:.3f} <= {MIN_ICIR}"
        elif twin:
            reason[name] = f"dropped: correlation {correlation.loc[name, twin[0]]:.2f} with {twin[0]} (higher ICIR)"
        else:
            kept.append(name)
            reason[name] = "kept"
    return kept, pd.Series(reason)


def decision_matrix(signal_tables, date, stocks, names):
    """stocks × signals on one date: z-scores, sign-adjusted so that higher is always better."""
    m = pd.DataFrame({name: signal_tables[name].loc[date, stocks] for name in names})
    z = (m - m.mean()) / m.std()
    return z * pd.Series({name: DIRECTIONS[name] for name in names})


# ============================================================================ step 1: signal weights
# Best-Worst Method judgement, written down from the literature (not from our data): how strong and
# how well-established each effect is for large, liquid stocks, on a 1–9 scale.
# Low volatility and momentum are the most robust anomalies; illiquidity matters little among the
# 100 largest (and most liquid) stocks; our own open signal is untested, so it gets a low score.
IMPORTANCE = {"volatility": 9, "momentum": 8, "high_52w": 6, "beta": 6, "max_return": 5,
              "reversal": 4, "open_vs_med21": 3, "illiquidity": 2}


def bwm_comparisons(names):
    """
    The two BWM comparison vectors for the chosen signals, read off the importance scores:
    best = highest score, worst = lowest;  a_best,j = 1 + (s_best − s_j),  a_j,worst = 1 + (s_j − s_worst).
    """
    s = pd.Series({n: IMPORTANCE[n] for n in names})
    best, worst = s.idxmax(), s.idxmin()
    return best, worst, 1 + s[best] - s, 1 + s - s[worst]


def bwm_weights(names):
    """
    Linear Best-Worst Method (Rezaei 2016): find weights w and the smallest ξ such that
        |w_best − a_best,j · w_j| ≤ ξ   and   |w_j − a_j,worst · w_worst| ≤ ξ   for every j,
        Σ w = 1,  w ≥ 0.
    ξ measures how inconsistent the judgement is (0 = perfectly consistent).
    Variables x = (w_1 … w_n, ξ); every |…| ≤ ξ becomes two linear inequalities.
    """
    best, worst, best_to_others, others_to_worst = bwm_comparisons(names)
    n, b, k = len(names), names.index(best), names.index(worst)
    rows = []
    for j, name in enumerate(names):
        row = np.zeros(n)                                    # w_best − a_bj·w_j
        row[b] += 1
        row[j] -= best_to_others[name]
        row2 = np.zeros(n)                                   # w_j − a_jw·w_worst
        row2[j] += 1
        row2[k] -= others_to_worst[name]
        for r in (row, row2):
            rows += [np.r_[r, -1], np.r_[-r, -1]]            # r·w − ξ ≤ 0  and  −r·w − ξ ≤ 0
    cost = np.r_[np.zeros(n), 1]                             # minimise ξ
    solution = linprog(cost, A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                       A_eq=[np.r_[np.ones(n), 0]], b_eq=[1], bounds=[(0, None)] * (n + 1))
    return pd.Series(solution.x[:n], index=names), solution.x[n]


def critic_weights(matrix):
    """
    CRITIC: a signal gets more weight if it varies a lot across stocks (std) and disagrees
    with the other signals (1 − correlation):  w_j ∝ std_j × Σ_k (1 − r_jk).
    """
    x = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    information = x.std() * (1 - x.corr()).sum()
    return information / information.sum()


# ============================================================================ step 2: rank the stocks
def weighted_sum(matrix, w):
    """Score = Σ_j w_j · z_j."""
    return matrix @ w


def topsis(matrix, w):
    """
    TOPSIS: scale each signal to 0–1, normalise, multiply by its weight; then measure each
    stock's distance to the IDEAL stock (best on every signal, d+) and to the WORST (d−).
    Score = d− / (d+ + d−): 1 = the ideal stock, 0 = the worst.
    """
    x = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    v = x / np.sqrt((x ** 2).sum()) * w
    d_best = np.sqrt(((v - v.max()) ** 2).sum(axis=1))
    d_worst = np.sqrt(((v - v.min()) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst)


SCHEMES = ["CRITIC + weighted sum", "BWM + TOPSIS"]


def scores(scheme, matrix, bwm):
    """One score per stock (higher = better) for one of the two MCDM schemes."""
    if scheme == "CRITIC + weighted sum":
        return weighted_sum(matrix, critic_weights(matrix))
    return topsis(matrix, bwm)
