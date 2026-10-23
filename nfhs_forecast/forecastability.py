"""Per-district noise ceiling and forecastability (Task C3; DECISIONS D34).

Central claim of the project: forecastability is a property of the indicator
AND the district, not only of the model. For one method's predictions:

  v_i            typical squared change of indicator i = MSE(persistence) over districts
  noise_share    se_true^2 / v_i      (share of a typical change that is pure sampling noise)
  norm_sq_err    (y_true - y_pred)^2 / v_i
  district d:    ceiling_d = 1 - mean_i noise_share_{d,i}     (best achievable normalised skill)
                 skill_d   = 1 - mean_i norm_sq_err_{d,i}      (achieved normalised skill)
                 gap_d     = ceiling_d - skill_d
A two-way additive decomposition of log(norm_sq_err) into district and
indicator effects reports how much of the variation in forecast error each
factor explains (sequential and partial R^2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _norm(preds: pd.DataFrame, method: str) -> pd.DataFrame:
    p = preds.dropna(subset=["y_true", "y_prev", "y_pred", "se_true"])
    base = p[p.method == "persistence"].assign(e2=lambda d: (d.y_true - d.y_prev) ** 2)
    v = base.groupby("indicator_id").e2.mean().rename("v")
    m = p[p.method == method].merge(v, left_on="indicator_id", right_index=True)
    m = m.assign(noise_share=m.se_true ** 2 / m.v, norm_sq_err=(m.y_true - m.y_pred) ** 2 / m.v)
    return m


def district_table(preds: pd.DataFrame, method: str, min_indicators: int = 10) -> pd.DataFrame:
    """One row per district: noise ceiling, achieved normalised skill, gap, n indicators."""
    m = _norm(preds, method)
    g = m.groupby("district_uid").agg(state=("state", "first"), n_indicators=("indicator_id", "size"),
                                      mean_noise_share=("noise_share", "mean"),
                                      mean_norm_sq_err=("norm_sq_err", "mean"))
    g["ceiling"] = 1 - g.mean_noise_share
    g["skill"] = 1 - g.mean_norm_sq_err
    g["gap"] = g.ceiling - g.skill
    g = g[g.n_indicators >= min_indicators]
    return g.reset_index().sort_values("skill", ascending=False).reset_index(drop=True)


def indicator_table(preds: pd.DataFrame, method: str) -> pd.DataFrame:
    m = _norm(preds, method)
    g = m.groupby("indicator_id").agg(n=("district_uid", "size"), mean_noise_share=("noise_share", "mean"),
                                      mean_norm_sq_err=("norm_sq_err", "mean"))
    g["ceiling"] = 1 - g.mean_noise_share
    g["skill"] = 1 - g.mean_norm_sq_err
    return g.reset_index().sort_values("skill", ascending=False).reset_index(drop=True)


def decompose(preds: pd.DataFrame, method: str, eps: float = 1e-3, iters: int = 50) -> dict:
    """Additive model log(norm_sq_err + eps) = mu + a_district + b_indicator (backfitting).

    Returns R^2 of indicator-only, district-only and the additive model, and
    the partial (unique) share of each factor.
    """
    m = _norm(preds, method)
    y = np.log(m.norm_sq_err.to_numpy() + eps)
    d = m.district_uid.to_numpy()
    i = m.indicator_id.to_numpy()
    tss = float(np.sum((y - y.mean()) ** 2))

    def r2(fit):
        return 1 - float(np.sum((y - fit) ** 2)) / tss if tss > 0 else np.nan

    s = pd.Series(y)
    fit_i = s.groupby(i).transform("mean").to_numpy()
    fit_d = s.groupby(d).transform("mean").to_numpy()
    a = np.zeros_like(y)
    b = np.zeros_like(y)
    mu = y.mean()
    for _ in range(iters):
        b = pd.Series(y - mu - a).groupby(i).transform("mean").to_numpy()
        a = pd.Series(y - mu - b).groupby(d).transform("mean").to_numpy()
    full = r2(mu + a + b)
    ri, rd = r2(fit_i), r2(fit_d)
    return {"method": method, "n": int(len(y)), "districts": int(len(set(d))), "indicators": int(len(set(i))),
            "r2_indicator_only": ri, "r2_district_only": rd, "r2_additive": full,
            "unique_indicator": full - rd, "unique_district": full - ri}


def plot_map(district_values: pd.Series, codes: pd.Series, geojson_path, path, title: str,
             label: str, cmap: str = "viridis") -> int:
    """Choropleth on Census-2011 polygons. district_values: index district_uid; codes: uid -> census code.

    Several units sharing one 2011 polygon are averaged; units without a code
    are not drawn (reported in the caption). Returns the number of polygons coloured.
    """
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from .spatial import _rings, norm_code
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)
    by_code = (pd.DataFrame({"v": district_values, "code": codes.reindex(district_values.index).map(norm_code)})
               .dropna().groupby("code").v.mean())
    polys, vals, empty = [], [], []
    for feat in gj["features"]:
        code = norm_code(feat["properties"].get("censuscode"))
        for ring in _rings(feat["geometry"]):
            xy = np.asarray(ring)[:, :2]
            if code in by_code.index:
                polys.append(xy)
                vals.append(by_code[code])
            else:
                empty.append(xy)
    fig, ax = plt.subplots(figsize=(7.5, 8.5), dpi=150)
    ax.add_collection(PolyCollection(empty, facecolor="#eeeeee", edgecolor="#cccccc", linewidths=0.2))
    pc = PolyCollection(polys, array=np.asarray(vals), cmap=cmap, edgecolor="white", linewidths=0.15)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(pc, ax=ax, shrink=0.6, label=label)
    missing = int(district_values.index.difference(codes.dropna().index).size)
    ax.set_title(title, fontsize=10, loc="left")
    fig.text(0.01, 0.01, f"Census-2011 polygons; grey = no value. {missing} units without a 2011 code not drawn.",
             fontsize=7, color="#555555")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return len(by_code)
