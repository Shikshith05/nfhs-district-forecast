"""Simple baselines that predict round t+1 from round t, one indicator at a time.

All functions work on aligned 1-D arrays/Series over districts. The "fit_*"
functions only ever see training pairs (rounds 4 -> 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(y_prev):
    """Forecast = last observed value."""
    return np.asarray(y_prev, dtype=float).copy()


# ---------------------------------------------------------------------
# State trend
# ---------------------------------------------------------------------
def fit_state_change(y_prev, y_next, states) -> tuple[dict, float]:
    """Mean district change per state (and nationally) on training pairs."""
    d = pd.DataFrame({"chg": np.asarray(y_next, float) - np.asarray(y_prev, float),
                      "state": np.asarray(states)}).dropna(subset=["chg"])
    by_state = d.groupby("state").chg.mean().to_dict()
    national = float(d.chg.mean()) if len(d) else 0.0
    return by_state, national


def state_trend(y_prev, states, state_change: dict, national_change: float,
                damping: float = 1.0):
    """Forecast = district value + its state's mean change (national mean if the state is unseen)."""
    chg = np.array([state_change.get(s, national_change) for s in np.asarray(states)], float)
    return np.asarray(y_prev, float) + damping * chg


def state_change_leave_one_out(y_prev, y_next, states):
    """For each district, its state's mean change computed WITHOUT that district.

    Uses target-round values of the other districts in the same state, so it
    is an oracle reference, not an honest forecast (DECISIONS D17).
    """
    d = pd.DataFrame({"chg": np.asarray(y_next, float) - np.asarray(y_prev, float),
                      "state": np.asarray(states)})
    ok = d.chg.notna()
    s = d.chg.where(ok, 0.0).groupby(d.state).transform("sum")
    n = ok.groupby(d.state).transform("sum")
    own = d.chg.where(ok, 0.0)
    loo_n = n - ok.astype(int)
    out = (s - own) / loo_n.replace(0, np.nan)
    national = d.chg.mean()
    return out.fillna(national).to_numpy()


# ---------------------------------------------------------------------
# Regression to the mean
# ---------------------------------------------------------------------
def fit_rtm(y_prev, y_next) -> tuple[float, float]:
    """OLS y_next = a + b * y_prev on training pairs. b < 1 means regression to the mean."""
    x = np.asarray(y_prev, float)
    y = np.asarray(y_next, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.var(x) == 0:
        return float(np.nanmean(y - x)) if len(x) else 0.0, 1.0
    b = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    a = float(y.mean() - b * x.mean())
    return a, b


def rtm(y_prev, a: float, b: float):
    return a + b * np.asarray(y_prev, float)


# ---------------------------------------------------------------------
# Empirical-Bayes shrinkage toward the state mean
# ---------------------------------------------------------------------
def eb_shrinkage(y_prev, se_prev, states, return_weights: bool = False):
    """Shrink each district's round-t value toward its state mean using its SE.

    theta_d = m_s + B_d (y_d - m_s),  B_d = tau^2 / (tau^2 + se_d^2)

    m_s is the state mean of round-t values; tau^2 (true between-district
    variance within states) is estimated by the method of moments, pooled
    over states: var(y - m_s) - mean(se^2), floored at 5% of var(y - m_s).
    The forecast for round t+1 is theta_d (denoised persistence). Uses only
    round-t information.
    """
    y = pd.Series(np.asarray(y_prev, float))
    se = pd.Series(np.asarray(se_prev, float))
    st = pd.Series(np.asarray(states))
    ok = y.notna() & se.notna()
    m = y.where(ok).groupby(st).transform("mean")
    resid = (y - m).where(ok)
    # states with a single district have resid == 0 and carry no information on tau^2
    n_state = ok.groupby(st).transform("sum")
    use = ok & (n_state > 1)
    v = float(np.nanvar(resid[use], ddof=1)) if use.sum() > 2 else 0.0
    tau2 = max(v - float(np.nanmean(se[use] ** 2)) if use.any() else 0.0, 0.05 * v)
    B = tau2 / (tau2 + se ** 2) if tau2 > 0 else pd.Series(np.zeros(len(y)))
    theta = m + B * (y - m)
    theta = theta.where(ok, y)  # no SE -> leave the value unchanged
    if return_weights:
        return theta.to_numpy(), B.to_numpy(), tau2
    return theta.to_numpy()
