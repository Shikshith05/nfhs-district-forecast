"""Evaluation: leave-one-state-out CV on NFHS-4 -> NFHS-5, metrics, bootstrap CIs, plot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines as bl
from .models import (FeatureSpec, Transition, assert_no_round6, build_transition, fit_methods,
                     predict_methods)

METHOD_ORDER = ["persistence", "state_trend", "rtm", "eb_shrinkage", "elastic_net", "lightgbm",
                "state_trend_oracle"]
#: Methods that use only information available at the forecast origin.
HONEST_METHODS = [m for m in METHOD_ORDER if m != "state_trend_oracle"]
METHOD_LABELS = {
    "persistence": "Persistence", "state_trend": "State trend", "rtm": "Regression to mean",
    "eb_shrinkage": "EB shrinkage", "elastic_net": "Elastic net", "lightgbm": "LightGBM",
    "state_trend_oracle": "State trend (oracle)",
}


def feature_spec(trans: Transition, indicators) -> FeatureSpec:
    return FeatureSpec(list(indicators), sorted(trans.meta.state.unique()),
                       sorted(trans.meta.region.unique()))


def attach_truth(preds: pd.DataFrame, trans: Transition) -> pd.DataFrame:
    def stack(df, name):
        s = df.stack(future_stack=True)
        s.index.names = ["district_uid", "indicator_id"]
        return s.rename(name).reset_index()
    truth = stack(trans.prev, "y_prev").merge(stack(trans.next, "y_true"),
                                              on=["district_uid", "indicator_id"])
    truth = truth.merge(stack(trans.se_next, "se_true"), on=["district_uid", "indicator_id"])
    out = preds.merge(truth, on=["district_uid", "indicator_id"], how="left")
    out["state"] = out.district_uid.str.split("|").str[0]
    return out


def loso_cv(train_panel: pd.DataFrame, indicators, uids, cfg: dict, verbose: bool = True
            ) -> pd.DataFrame:
    """Leave-one-state-out predictions for NFHS-4 -> NFHS-5.

    Per held-out state: fit state trend / RTM / elastic net / LightGBM on the
    other states, predict the held-out state. Persistence and EB shrinkage use
    round-4 data only and are computed on the full cross-section. The oracle
    state trend uses the held-out state's own round-5 values of the other
    districts (a reference, not a forecast).
    """
    assert_no_round6(train_panel, "loso_cv")
    trans = build_transition(train_panel, 4, 5, indicators, uids, cfg)
    spec = feature_spec(trans, indicators)
    parts = []
    # EB on the full round-4 cross-section, at the default and sensitivity deffs
    deff0 = cfg["noise"]["deff"]
    for deff in sorted(set(cfg["noise"]["deff_sensitivity"]) | {deff0}):
        scale = np.sqrt(deff / deff0)
        for ind in indicators:
            p = bl.eb_shrinkage(trans.prev[ind].to_numpy(), trans.se_prev[ind].to_numpy() * scale,
                                trans.meta.state)
            name = "eb_shrinkage" if deff == deff0 else f"eb_shrinkage_deff{deff:g}"
            parts.append(pd.DataFrame({"district_uid": trans.prev.index, "indicator_id": ind,
                                       "method": name, "y_pred": p}))
    states = sorted(trans.meta.state.unique())
    for i, st in enumerate(states, 1):
        test_uids = trans.meta.index[trans.meta.state == st]
        train_uids = trans.meta.index[trans.meta.state != st]
        fitted = fit_methods(trans.subset(train_uids), cfg, spec)
        test = trans.subset(test_uids)
        parts.append(predict_methods(fitted, test, cfg, include_eb=False))
        for ind in indicators:
            chg = bl.state_change_leave_one_out(test.prev[ind], test.next[ind], test.meta.state)
            parts.append(pd.DataFrame({"district_uid": test_uids, "indicator_id": ind,
                                       "method": "state_trend_oracle",
                                       "y_pred": np.clip(test.prev[ind].to_numpy() + chg, 0, 100)}))
        if verbose:
            print(f"  LOSO fold {i:2d}/{len(states)}: held out {st} ({len(test_uids)} districts)")
    preds = pd.concat(parts, ignore_index=True)
    preds.loc[preds.method.str.startswith("eb_"), "y_pred"] = preds.y_pred.clip(0, 100)
    return attach_truth(preds, trans)


# =====================================================================
# Metrics
# =====================================================================
def _common_support(preds: pd.DataFrame, methods) -> pd.DataFrame:
    p = preds[preds.method.isin(methods)].dropna(subset=["y_true", "y_prev"])
    full = p.groupby(["district_uid", "indicator_id"]).y_pred.apply(lambda s: s.notna().all())
    keep = full[full].index
    return p.set_index(["district_uid", "indicator_id"]).loc[keep].reset_index()


def _state_sums(p: pd.DataFrame) -> pd.DataFrame:
    p = p.assign(err=p.y_pred - p.y_true)
    p = p.assign(abs_err=p.err.abs(), sq_err=p.err ** 2, noise=p.se_true ** 2)
    return p.groupby(["state", "indicator_id", "method"]).agg(
        n=("err", "size"), sum_abs=("abs_err", "sum"), sum_sq=("sq_err", "sum"),
        sum_noise=("noise", "sum")).reset_index()


def _metrics_from_sums(n, sum_abs, sum_sq, sum_sq_base, sum_noise):
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = sum_abs / n
        mse = sum_sq / n
        skill = 1 - sum_sq / sum_sq_base
        noise_mse = sum_noise / n
        ceiling = 1 - sum_noise / sum_sq_base
    return mae, np.sqrt(mse), mse, skill, noise_mse, ceiling


def metric_table(preds: pd.DataFrame, methods=None, baseline: str = "persistence",
                 bootstrap_reps: int = 1000, ci_level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """MAE, RMSE, skill vs persistence, noise floor; per indicator and OVERALL, with
    95% CIs from a state-clustered bootstrap (states resampled with replacement)."""
    methods = list(methods or [m for m in preds.method.unique()])
    if baseline not in methods:
        methods.append(baseline)
    p = _common_support(preds, methods)
    sums = _state_sums(p)
    ind_sums = sums.copy()
    over = sums.groupby(["state", "method"])[["n", "sum_abs", "sum_sq", "sum_noise"]].sum().reset_index()
    over["indicator_id"] = "OVERALL"
    sums = pd.concat([ind_sums, over], ignore_index=True)

    states = sorted(sums.state.unique())
    keys = sums[["indicator_id", "method"]].drop_duplicates().sort_values(["indicator_id", "method"])
    keys = keys.reset_index(drop=True)
    def matrix(col):
        m = sums.pivot_table(index="state", columns=["indicator_id", "method"], values=col,
                             aggfunc="sum", fill_value=0.0)
        m = m.reindex(index=states, columns=pd.MultiIndex.from_frame(keys), fill_value=0.0)
        return m.to_numpy(float)
    N, A, S, Z = (matrix(c) for c in ("n", "sum_abs", "sum_sq", "sum_noise"))
    base_cols = np.array([keys.index[(keys.indicator_id == ind) & (keys.method == baseline)][0]
                          for ind in keys.indicator_id])

    def all_metrics(n, a, s, z):
        return _metrics_from_sums(n, a, s, s[..., base_cols], z)

    tot = [x.sum(axis=0) for x in (N, A, S, Z)]
    mae, rmse, mse, skill, noise_mse, ceiling = all_metrics(*tot)
    out = keys.copy()
    out["n"] = tot[0].astype(int)
    out["mae"], out["rmse"], out["mse"], out["skill"] = mae, rmse, mse, skill
    out["noise_mse"], out["skill_ceiling"] = noise_mse, ceiling

    if bootstrap_reps:
        rng = np.random.default_rng(seed)
        W = rng.multinomial(len(states), np.full(len(states), 1 / len(states)), size=bootstrap_reps)
        bm = all_metrics(W @ N, W @ A, W @ S, W @ Z)
        lo, hi = 100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2
        for name, arr in zip(["mae", "rmse"], bm[:2]):
            out[f"{name}_lo"], out[f"{name}_hi"] = np.nanpercentile(arr, lo, axis=0), np.nanpercentile(arr, hi, axis=0)
        out["skill_lo"], out["skill_hi"] = np.nanpercentile(bm[3], lo, axis=0), np.nanpercentile(bm[3], hi, axis=0)
    out["honest"] = ~out.method.eq("state_trend_oracle")
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    out = out.assign(_m=out.method.map(order).fillna(99), _o=out.indicator_id.eq("OVERALL"))
    out = out.sort_values(["_o", "indicator_id", "_m"]).drop(columns=["_m", "_o"])
    cols = ["indicator_id", "method", "honest", "n", "mae", "mae_lo", "mae_hi", "rmse", "rmse_lo",
            "rmse_hi", "skill", "skill_lo", "skill_hi", "noise_mse", "skill_ceiling"]
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def deff_sensitivity(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """EB skill and noise ceiling under the deff sensitivity range."""
    deff0 = cfg["noise"]["deff"]
    rows = []
    for deff in cfg["noise"]["deff_sensitivity"]:
        name = "eb_shrinkage" if deff == deff0 else f"eb_shrinkage_deff{deff:g}"
        sub = preds[preds.method.isin(["persistence", name])].copy()
        sub["se_true"] = sub.se_true * np.sqrt(deff / deff0)
        sub["method"] = sub.method.replace({name: "eb_shrinkage"})
        t = metric_table(sub, ["persistence", "eb_shrinkage"], bootstrap_reps=0)
        t = t[t.method == "eb_shrinkage"].assign(deff=deff)
        rows.append(t[["deff", "indicator_id", "mae", "rmse", "skill", "noise_mse", "skill_ceiling"]])
    return pd.concat(rows, ignore_index=True)


# =====================================================================
# Plot
# =====================================================================
def plot_skill(results: pd.DataFrame, path, title: str, labels: dict[str, str] | None = None,
               xmin: float = -1.0) -> None:
    """Small multiples: one panel per honest method, skill vs persistence with 95% CI per
    indicator. Filled dot = CI excludes 0; hollow = not distinguishable from persistence.
    Values below `xmin` are drawn at the edge with a left-pointing marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, band, mark = "#0b0b0b", "#52514e", "#e1dfda", "#f0efec", "#2a78d6"
    methods = [m for m in HONEST_METHODS if m != "persistence" and m in set(results.method)]
    r = results[results.method.isin(methods)].copy()
    best_ml = (r[r.method.isin(["elastic_net", "lightgbm"]) & (r.indicator_id != "OVERALL")]
               .groupby("indicator_id").skill.max().sort_values(ascending=False))
    order = ["OVERALL"] + list(best_ml.index)
    labels = labels or {}
    y = np.arange(len(order))
    hi = float(min(max(0.5, np.nanmax(r.get("skill_hi", r.skill)) + 0.05), 1.0))
    lowest = float(np.nanmin(r.get("skill_lo", r.skill)))
    xmin = max(xmin, min(-0.25, np.floor(lowest * 10) / 10 - 0.05))
    fig, axes = plt.subplots(1, len(methods), figsize=(2.0 * len(methods) + 2.2, 0.27 * len(order) + 1.5),
                             dpi=160, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        d = r[r.method == m].set_index("indicator_id").reindex(order)
        x = d.skill.to_numpy()
        lo_ci = d.skill_lo.to_numpy() if "skill_lo" in d else x
        hi_ci = d.skill_hi.to_numpy() if "skill_hi" in d else x
        sig = (lo_ci > 0) | (hi_ci < 0)
        ax.axhspan(-0.5, 0.5, color=band, zorder=0)
        ax.axvline(0, color=ink2, lw=0.9, zorder=1)
        ax.hlines(y, np.clip(lo_ci, xmin, hi), np.clip(hi_ci, xmin, hi), color=mark, lw=1.1,
                  alpha=0.55, zorder=2)
        inside = x >= xmin
        ax.scatter(x[inside & sig], y[inside & sig], s=24, color=mark, edgecolors="white",
                   linewidths=0.8, zorder=3)
        ax.scatter(x[inside & ~sig], y[inside & ~sig], s=24, facecolors="white", edgecolors=mark,
                   linewidths=1.1, zorder=3)
        ax.scatter(np.full((~inside).sum(), xmin), y[~inside], s=26, marker="<", color=mark, zorder=3)
        ax.set_xlim(xmin - 0.05, hi)
        ax.set_title(METHOD_LABELS[m], fontsize=8.5, color=ink, loc="left")
        ax.grid(axis="x", color=grid, lw=0.6, zorder=0)
        ax.tick_params(axis="x", labelsize=7, colors=ink2)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(y)
    first.set_yticklabels([labels.get(i, i) for i in order], fontsize=7.5, color=ink)
    first.invert_yaxis()
    fig.suptitle(title, fontsize=10, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Skill = 1 − MSE(method) / MSE(persistence); > 0 beats persistence. "
             "Lines: 95% CI, state-clustered bootstrap. Filled dot: CI excludes 0. "
             f"◄ = skill below {xmin:g}.", fontsize=6.8, color=ink2, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
