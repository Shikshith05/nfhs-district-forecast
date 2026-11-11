"""Conformal prediction intervals calibrated on NFHS-4 -> NFHS-5 residuals (Task C1; DECISIONS D34).

All functions take long prediction tables in the format of
outputs/predictions_nfhs4to5.parquet (calibration) and
outputs/predictions_nfhs5to6.parquet (target): one row per district_uid x
indicator_id x method with y_pred, y_prev, se_prev, and (to score) y_true.

Three pre-registered variants, each calibrated per indicator x method
(Mondrian by indicator):
  split      score |y - y_hat|; interval y_hat +/- q, q the
             ceil((n+1)(1-alpha))-th smallest calibration score;
  normalised score |y - y_hat| / s, s = sqrt(se_prev^2 + s0^2), with s0^2 the
             median squared calibration residual (a stabiliser). The width
             adapts to each district's origin sampling noise, which is known
             at the forecast origin;
  weighted   split scores re-weighted for covariate shift (Tibshirani et al.
             2019). w(x) = p(target | x) / p(calibration | x) is estimated by
             a logistic classifier on round-t inputs only (y_prev of the
             indicator, se_prev, region). The quantile is the weighted
             (1-alpha) quantile including the test point's own weight at +inf.
Coverage on the target = share of y_true inside the interval. Intervals never
use target outcomes: y_true is read only to measure coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VARIANTS = ("split", "normalised", "weighted")


def conformal_quantile(scores, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small."""
    s = np.sort(np.asarray(scores, float)[~np.isnan(scores)])
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if 1 <= k <= n else np.inf


def weighted_quantile(scores, weights, test_weight: float, alpha: float) -> float:
    """(1-alpha) quantile of sum_i p_i delta_{s_i} + p_test delta_{+inf}, p proportional to weights."""
    s = np.asarray(scores, float)
    w = np.asarray(weights, float)
    ok = ~(np.isnan(s) | np.isnan(w))
    s, w = s[ok], w[ok]
    order = np.argsort(s)
    s, w = s[order], w[order]
    total = w.sum() + test_weight
    if total <= 0:
        return np.inf
    cum = np.cumsum(w) / total
    idx = np.searchsorted(cum, 1 - alpha, side="left")
    return float(s[idx]) if idx < len(s) else np.inf


def _scale(se_prev, s0_sq: float):
    return np.sqrt(np.asarray(se_prev, float) ** 2 + s0_sq)


def shift_weights(cal_x: pd.DataFrame, tgt_x: pd.DataFrame, clip: float = 20.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Density-ratio weights p(target|x)/p(cal|x) from a logistic classifier on inputs only.

    Returns (weights for calibration rows, weights for target rows), clipped to [1/clip, clip].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = pd.concat([cal_x, tgt_x], ignore_index=True)
    X = pd.get_dummies(X, dtype=float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = np.r_[np.zeros(len(cal_x)), np.ones(len(tgt_x))]
    if len(np.unique(y)) < 2:
        return np.ones(len(cal_x)), np.ones(len(tgt_x))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    ratio = (p / (1 - p)) * (len(cal_x) / max(len(tgt_x), 1))
    ratio = np.clip(ratio, 1 / clip, clip)
    return ratio[:len(cal_x)], ratio[len(cal_x):]


def _inputs(df: pd.DataFrame, region: pd.Series | None) -> pd.DataFrame:
    x = pd.DataFrame({"y_prev": df.y_prev.to_numpy(), "se_prev": df.se_prev.to_numpy()})
    if region is not None:
        x["region"] = df.state.map(region).fillna("other").to_numpy()
    return x


def conformal_intervals(cal: pd.DataFrame, target: pd.DataFrame, alpha: float = 0.1,
                        variants=VARIANTS, methods=None, region: dict | None = None,
                        weight_clip: float = 20.0, seed: int = 0) -> pd.DataFrame:
    """Intervals for every target row, per variant; calibrated per indicator x method on `cal`.

    cal must have y_true (calibration residuals); target needs y_pred, y_prev,
    se_prev (and y_true only if coverage is to be measured afterwards).
    """
    methods = methods or sorted(set(cal.method) & set(target.method))
    region = pd.Series(region or {})
    out = []
    for (ind, m), t in target[target.method.isin(methods)].groupby(["indicator_id", "method"]):
        c = cal[(cal.indicator_id == ind) & (cal.method == m)].dropna(subset=["y_true", "y_pred"])
        if c.empty:
            continue
        r = (c.y_true - c.y_pred).abs().to_numpy()
        s0_sq = float(np.median(r ** 2))
        t = t.copy()
        if "split" in variants:
            q = conformal_quantile(r, alpha)
            out.append(t.assign(variant="split", lo=t.y_pred - q, hi=t.y_pred + q, q=q))
        if "normalised" in variants:
            q = conformal_quantile(r / _scale(c.se_prev, s0_sq), alpha)
            half = q * _scale(t.se_prev, s0_sq)
            out.append(t.assign(variant="normalised", lo=t.y_pred - half, hi=t.y_pred + half, q=q))
        if "weighted" in variants:
            wc, wt = shift_weights(_inputs(c, region if len(region) else None),
                                   _inputs(t, region if len(region) else None), weight_clip, seed)
            qs = np.array([weighted_quantile(r, wc, w, alpha) for w in wt])
            out.append(t.assign(variant="weighted", lo=t.y_pred - qs, hi=t.y_pred + qs, q=qs))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["width"] = res.hi - res.lo
    if "y_true" in res.columns:
        res["covered"] = np.where(res.y_true.isna(), np.nan, (res.y_true >= res.lo) & (res.y_true <= res.hi))
    return res


def coverage_table(intervals: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Coverage and mean width per variant x method x indicator, plus OVERALL rows."""
    d = intervals.dropna(subset=["covered"])
    d = d.assign(covered=d.covered.astype(float), finite=np.isfinite(d.width))
    g = d.groupby(["variant", "method", "indicator_id"]).agg(
        n=("covered", "size"), coverage=("covered", "mean"),
        mean_width=("width", lambda w: float(np.mean(w[np.isfinite(w)])) if np.isfinite(w).any() else np.inf))
    o = d.groupby(["variant", "method"]).agg(
        n=("covered", "size"), coverage=("covered", "mean"),
        mean_width=("width", lambda w: float(np.mean(w[np.isfinite(w)])) if np.isfinite(w).any() else np.inf))
    o["indicator_id"] = "OVERALL"
    out = pd.concat([g.reset_index(), o.reset_index()], ignore_index=True)
    out["nominal"] = 1 - alpha
    # binomial 95% band for the coverage of n exchangeable points
    out["coverage_se"] = np.sqrt(out.coverage * (1 - out.coverage) / out.n)
    return out.sort_values(["variant", "method", "indicator_id"]).reset_index(drop=True)
