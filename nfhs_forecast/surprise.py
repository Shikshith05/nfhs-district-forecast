"""FDR-controlled "surprise" districts (Task C2; DECISIONS D34).

A district x indicator is a surprise if its observed value is far from the
frozen forecast relative to BOTH the forecast-error variance and the sampling
variance of the observed value:

    z = (y_true - y_pred) / sqrt(sigma2_f + se_true^2)

sigma2_f (per indicator x method) is the forecast-error variance net of
sampling noise, estimated on the calibration residuals (NFHS-4 -> NFHS-5
LOSO): max(var(resid) - mean(se_true^2), floor * var(resid)). Two p-values:
  p_normal     2 * (1 - Phi(|z|))
  p_conformal  (1 + #{calibration |z_cal| >= |z|}) / (n_cal + 1)
               (distribution-free; valid for BH, Bates et al. 2023)
Benjamini-Hochberg is applied over all district x indicator pairs of one
method (family = all tests), at q = surprise.fdr_q. Naive top-k (largest
|residual|, k = number of BH discoveries, at least `naive_k_min`) is reported
for comparison: it ignores noise and so favours small-sample districts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .evaluate import bh_adjust


def forecast_error_variance(cal: pd.DataFrame, floor_share: float = 0.05) -> pd.DataFrame:
    """sigma2_f per indicator x method from calibration residuals (net of sampling noise)."""
    c = cal.dropna(subset=["y_true", "y_pred", "se_true"]).assign(r=lambda d: d.y_true - d.y_pred)
    g = c.groupby(["indicator_id", "method"]).agg(var_resid=("r", "var"), mean_resid=("r", "mean"),
                                                  noise=("se_true", lambda s: float(np.mean(s ** 2))),
                                                  n_cal=("r", "size"))
    g["sigma2_f"] = np.maximum(g.var_resid - g.noise, floor_share * g.var_resid)
    return g.reset_index()


def standardise(df: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(sig[["indicator_id", "method", "sigma2_f"]], on=["indicator_id", "method"], how="left")
    out["resid"] = out.y_true - out.y_pred
    out["scale"] = np.sqrt(out.sigma2_f + out.se_true ** 2)
    out["z"] = out.resid / out.scale
    return out


def detect_surprises(cal: pd.DataFrame, target: pd.DataFrame, method: str, q: float = 0.1,
                     floor_share: float = 0.05, naive_k_min: int = 10) -> tuple[pd.DataFrame, dict]:
    """BH-FDR surprises for one method. Returns (per-pair table, summary)."""
    cal_m = cal[cal.method == method]
    tgt = target[(target.method == method)].dropna(subset=["y_true", "y_pred", "se_true"]).copy()
    sig = forecast_error_variance(cal_m, floor_share)
    t = standardise(tgt, sig).dropna(subset=["z"])
    c = standardise(cal_m.dropna(subset=["y_true", "y_pred", "se_true"]), sig).dropna(subset=["z"])
    t["p_normal"] = 2 * (1 - norm.cdf(np.abs(t.z)))
    cal_abs = {ind: np.sort(np.abs(g.z.to_numpy())) for ind, g in c.groupby("indicator_id")}

    def p_conf(row):
        s = cal_abs.get(row.indicator_id)
        if s is None or len(s) == 0:
            return np.nan
        ge = len(s) - np.searchsorted(s, abs(row.z), side="left")
        return (1 + ge) / (len(s) + 1)
    t["p_conformal"] = [p_conf(r) for r in t.itertuples()]
    t["q_normal"] = bh_adjust(t.p_normal)
    t["q_conformal"] = bh_adjust(t.p_conformal)
    t["surprise"] = t.q_normal < q
    t["surprise_conformal"] = t.q_conformal < q
    t["direction"] = np.where(t.resid > 0, "higher than forecast", "lower than forecast")
    k = max(int(t.surprise.sum()), naive_k_min)
    top = t.resid.abs().rank(ascending=False, method="first") <= k
    t["naive_topk"] = top
    summary = {
        "method": method, "q": q, "tests": int(len(t)), "surprises_normal": int(t.surprise.sum()),
        "surprises_conformal": int(t.surprise_conformal.sum()), "naive_k": k,
        "overlap_naive_vs_bh": int((t.naive_topk & t.surprise).sum()),
        "naive_topk_mean_se_true": float(t.loc[t.naive_topk, "se_true"].mean()),
        "bh_mean_se_true": float(t.loc[t.surprise, "se_true"].mean()) if t.surprise.any() else np.nan,
        "all_mean_se_true": float(t.se_true.mean()),
        "calibration_z_sd": float(c.z.std()), "target_z_sd": float(t.z.std()),
    }
    cols = ["district_uid", "state", "indicator_id", "method", "y_prev", "y_pred", "y_true", "resid", "se_true",
            "sigma2_f", "scale", "z", "p_normal", "q_normal", "p_conformal", "q_conformal", "surprise",
            "surprise_conformal", "naive_topk", "direction"]
    extra = [c for c in ("unit_type", "stable_unit", "parent_uid") if c in t.columns]
    t = t[cols + extra].sort_values("p_normal").reset_index(drop=True)
    return t, summary
