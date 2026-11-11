"""Evaluation: CV on NFHS-4 -> NFHS-5, metrics, bootstrap CIs, paired tests, plot.

Primary validation is leave-one-state-out (LOSO) CV. A secondary seen-state
CV (districts held out, states seen) is optimistic and only a reference
(DECISIONS D32). The same metric / test functions score NFHS-5 -> NFHS-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from . import baselines as bl
from .models import (FEATURE_SETS, ML_METHODS, FeatureSpec, Transition, assert_no_round6, build_transition,
                     fh_matrix, fit_methods, ml_method_name, predict_methods)

BASELINE_METHODS = ["persistence", "national_trend", "state_trend", "rtm", "fay_herriot", "fh_rtm"]
PRIMARY_ML = list(ML_METHODS)
VARIANT_ML = [ml_method_name(m, fs) for fs in FEATURE_SETS if fs != "full" for m in ML_METHODS]
METHOD_ORDER = BASELINE_METHODS + PRIMARY_ML + VARIANT_ML + ["state_trend_oracle"]
#: Methods that use only information available at the forecast origin.
HONEST_METHODS = [m for m in METHOD_ORDER if m != "state_trend_oracle"]
#: Methods shown in the skill plot (ablation variants are in the tables only).
PLOT_METHODS = ["state_trend", "rtm", "fay_herriot", "fh_rtm", "elastic_net", "lightgbm"]
#: Pre-registered paired comparisons (D32): (method, reference).
PAIRED_COMPARISONS = [("elastic_net", "rtm"), ("lightgbm", "rtm"),
                      ("elastic_net__nostate", "rtm"), ("lightgbm__nostate", "rtm")]
METHOD_LABELS = {
    "persistence": "Persistence", "national_trend": "National trend", "state_trend": "State trend",
    "rtm": "Regression to mean", "fay_herriot": "Fay-Herriot", "fh_rtm": "FH + RTM",
    "eb_shrinkage": "EB shrinkage", "elastic_net": "Elastic net", "lightgbm": "LightGBM",
    "state_trend_oracle": "State trend (oracle)",
}


def feature_spec(trans: Transition, indicators, feature_sets=None, adjacency=None) -> FeatureSpec:
    return FeatureSpec(list(indicators), sorted(trans.meta.state.unique()),
                       sorted(trans.meta.region.unique()), list(feature_sets or ["full"]), adjacency)


def attach_truth(preds: pd.DataFrame, trans: Transition) -> pd.DataFrame:
    def stack(df, name):
        s = df.stack(future_stack=True)
        s.index.names = ["district_uid", "indicator_id"]
        return s.rename(name).reset_index()
    truth = stack(trans.prev, "y_prev").merge(stack(trans.next, "y_true"),
                                              on=["district_uid", "indicator_id"])
    truth = truth.merge(stack(trans.se_prev, "se_prev"), on=["district_uid", "indicator_id"])
    truth = truth.merge(stack(trans.se_next, "se_true"), on=["district_uid", "indicator_id"])
    out = preds.merge(truth, on=["district_uid", "indicator_id"], how="left")
    out["state"] = out.district_uid.str.split("|").str[0]
    return out


def annotate_units(preds: pd.DataFrame, units: pd.DataFrame | None, stable_units) -> pd.DataFrame:
    """Add unit_type, stable_unit and parent_uid (bootstrap cluster) to a predictions table."""
    out = preds.copy()
    if units is not None and len(units):
        u = units.set_index("unit_uid")
        out["unit_type"] = out.district_uid.map(u.unit_type).fillna("identity")
        out["parent_uid"] = out.district_uid.map(u.parent_uid).fillna(out.district_uid)
    else:
        out["unit_type"] = "identity"
        out["parent_uid"] = out.district_uid
    out["stable_unit"] = out.district_uid.isin(set(stable_units))
    return out


# =====================================================================
# Cross-validation on NFHS-4 -> NFHS-5
# =====================================================================
def _denoiser_preds(trans: Transition, indicators, cfg: dict) -> list[pd.DataFrame]:
    """Fay-Herriot on the full round-4 cross-section, at the default and sensitivity deffs."""
    parts = []
    deff0 = cfg["noise"]["deff"]
    for deff in sorted(set(cfg["noise"]["deff_sensitivity"]) | {deff0}):
        fh = trans.fh if deff == deff0 else fh_matrix(trans.prev[indicators], trans.se_prev[indicators],
                                                       trans.meta.state, np.sqrt(deff / deff0))
        name = "fay_herriot" if deff == deff0 else f"fay_herriot_deff{deff:g}"
        for ind in indicators:
            parts.append(pd.DataFrame({"district_uid": trans.prev.index, "indicator_id": ind,
                                       "method": name, "y_pred": fh[ind].clip(0, 100).to_numpy()}))
    return parts


def _fit_predict_fold(trans: Transition, spec: FeatureSpec, cfg: dict, train_uids, test_uids,
                      feature_sets, oracle: bool) -> pd.DataFrame:
    fitted = fit_methods(trans.subset(train_uids), cfg, spec, feature_sets=feature_sets)
    test = trans.subset(test_uids)
    parts = [predict_methods(fitted, test, cfg, include_eb=False)]
    if oracle:
        for ind in spec.indicators:
            chg = bl.state_change_leave_one_out(test.prev[ind], test.next[ind], test.meta.state)
            parts.append(pd.DataFrame({"district_uid": list(test_uids), "indicator_id": ind,
                                       "method": "state_trend_oracle",
                                       "y_pred": np.clip(test.prev[ind].to_numpy() + chg, 0, 100)}))
    return pd.concat(parts, ignore_index=True)


def loso_cv(train_panel: pd.DataFrame, indicators, uids, cfg: dict, verbose: bool = True,
            adjacency: dict | None = None, feature_sets=None) -> pd.DataFrame:
    """Leave-one-state-out predictions for NFHS-4 -> NFHS-5.

    Per held-out state: fit trends / RTM / FH-RTM / ML models on the other
    states and predict the held-out state. Persistence and Fay-Herriot use
    round-4 data only and are computed on the full cross-section, as are the
    spatial-neighbour features. The oracle state trend uses the held-out
    state's own round-5 values of the other districts (a reference, not a
    forecast). Folds run in parallel (evaluation.n_jobs); results do not
    depend on n_jobs.
    """
    assert_no_round6(train_panel, "loso_cv")
    sets = list(feature_sets or cfg["models"].get("feature_sets", ["full"]))
    trans = build_transition(train_panel, 4, 5, indicators, uids, cfg, adjacency=adjacency)
    spec = feature_spec(trans, indicators, sets, adjacency)
    states = sorted(trans.meta.state.unique())
    folds = [(trans.meta.index[trans.meta.state != st], trans.meta.index[trans.meta.state == st])
             for st in states]
    n_jobs = cfg.get("evaluation", {}).get("n_jobs", 1)
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_predict_fold)(trans, spec, cfg, tr, te, sets, True)
                                  for tr, te in folds)
    if verbose:
        for st, (_, te) in zip(states, folds):
            print(f"  LOSO fold: held out {st} ({len(te)} districts)")
    preds = pd.concat(_denoiser_preds(trans, indicators, cfg) + out, ignore_index=True)
    return attach_truth(preds, trans)


def seen_state_cv(train_panel: pd.DataFrame, indicators, uids, cfg: dict, groups: pd.Series | None = None,
                  adjacency: dict | None = None, n_splits: int = 10) -> pd.DataFrame:
    """Secondary CV: districts held out (GroupKFold by parent unit), every state seen.

    Exercises the state dummies as the NFHS-6 test does, but shares the
    transition's state shocks between training and held-out districts, so it
    is OPTIMISTIC (DECISIONS D32). Fits only the FULL and `nostate` ML sets.
    """
    assert_no_round6(train_panel, "seen_state_cv")
    sets = ["full", "nostate"]
    trans = build_transition(train_panel, 4, 5, indicators, uids, cfg, adjacency=adjacency)
    spec = feature_spec(trans, indicators, sets, adjacency)
    g = (groups.reindex(trans.prev.index).fillna(pd.Series(trans.prev.index, index=trans.prev.index))
         if groups is not None else pd.Series(trans.prev.index, index=trans.prev.index))
    idx = trans.prev.index
    folds = [(idx[tr], idx[te]) for tr, te in GroupKFold(n_splits=n_splits).split(idx, groups=g.to_numpy())]
    n_jobs = cfg.get("evaluation", {}).get("n_jobs", 1)
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_predict_fold)(trans, spec, cfg, tr, te, sets, False)
                                  for tr, te in folds)
    fh = [d for d in _denoiser_preds(trans, indicators, cfg) if d.method.iloc[0] == "fay_herriot"]
    preds = pd.concat(fh + out, ignore_index=True)
    return attach_truth(preds, trans)


# =====================================================================
# Metrics
# =====================================================================
def _common_support(preds: pd.DataFrame, methods) -> pd.DataFrame:
    p = preds[preds.method.isin(methods)].dropna(subset=["y_true", "y_prev"])
    full = p.groupby(["district_uid", "indicator_id"]).y_pred.apply(
        lambda s: s.notna().all() and len(s) == len(set(methods)))
    keep = full[full].index
    return p.set_index(["district_uid", "indicator_id"]).loc[keep].reset_index()


def _state_sums(p: pd.DataFrame, cluster: str = "state") -> pd.DataFrame:
    p = p.assign(err=p.y_pred - p.y_true)
    p = p.assign(abs_err=p.err.abs(), sq_err=p.err ** 2, noise=p.se_true ** 2)
    return p.groupby([cluster, "indicator_id", "method"]).agg(
        n=("err", "size"), sum_abs=("abs_err", "sum"), sum_sq=("sq_err", "sum"),
        sum_noise=("noise", "sum")).reset_index().rename(columns={cluster: "cluster"})


def _metrics_from_sums(n, sum_abs, sum_sq, sum_sq_base, sum_noise):
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = sum_abs / n
        mse = sum_sq / n
        skill = 1 - sum_sq / sum_sq_base
        noise_mse = sum_noise / n
        ceiling = 1 - sum_noise / sum_sq_base
    return mae, np.sqrt(mse), mse, skill, noise_mse, ceiling


def _cluster_matrices(preds: pd.DataFrame, methods, cluster: str = "state"):
    """Per-cluster sums as matrices (clusters x [indicator, method] keys), incl. OVERALL keys."""
    p = _common_support(preds, methods)
    sums = _state_sums(p, cluster)
    over = sums.groupby(["cluster", "method"])[["n", "sum_abs", "sum_sq", "sum_noise"]].sum().reset_index()
    over["indicator_id"] = "OVERALL"
    sums = pd.concat([sums, over], ignore_index=True)
    clusters = sorted(sums.cluster.unique())
    keys = (sums[["indicator_id", "method"]].drop_duplicates().sort_values(["indicator_id", "method"])
            .reset_index(drop=True))

    def matrix(col):
        m = sums.pivot_table(index="cluster", columns=["indicator_id", "method"], values=col,
                             aggfunc="sum", fill_value=0.0)
        m = m.reindex(index=clusters, columns=pd.MultiIndex.from_frame(keys), fill_value=0.0)
        return m.to_numpy(float)
    return keys, clusters, [matrix(c) for c in ("n", "sum_abs", "sum_sq", "sum_noise")]


def _bootstrap_weights(n_clusters: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=reps)


def metric_table(preds: pd.DataFrame, methods=None, baseline: str = "persistence",
                 bootstrap_reps: int = 1000, ci_level: float = 0.95, seed: int = 0,
                 cluster: str = "state") -> pd.DataFrame:
    """MAE, RMSE, skill vs `baseline`, noise floor; per indicator, OVERALL (pooled squared
    errors) and MEAN (unweighted mean over indicators), with CIs from a cluster bootstrap
    (states resampled with replacement by default)."""
    present = set(preds.method.unique())
    methods = [m for m in (methods or list(present)) if m in present]
    if baseline not in methods:
        methods.append(baseline)
    keys, clusters, (N, A, S, Z) = _cluster_matrices(preds, methods, cluster)
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

    # MEAN: unweighted mean over indicators of each per-indicator metric
    ind_cols = {m: np.where((keys.method == m) & (keys.indicator_id != "OVERALL"))[0] for m in methods}
    mean_rows = [{"indicator_id": "MEAN", "method": m, "n": int(out.n.iloc[c].sum()),
                  "mae": float(np.mean(mae[c])), "rmse": float(np.mean(rmse[c])), "mse": float(np.mean(mse[c])),
                  "skill": float(np.mean(skill[c])), "noise_mse": float(np.mean(noise_mse[c])),
                  "skill_ceiling": float(np.mean(ceiling[c]))} for m, c in ind_cols.items() if len(c)]

    if bootstrap_reps:
        W = _bootstrap_weights(len(clusters), bootstrap_reps, seed)
        bm = all_metrics(W @ N, W @ A, W @ S, W @ Z)
        lo, hi = 100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2
        for name, arr in zip(["mae", "rmse"], bm[:2]):
            out[f"{name}_lo"], out[f"{name}_hi"] = np.nanpercentile(arr, lo, axis=0), np.nanpercentile(arr, hi, axis=0)
        out["skill_lo"], out["skill_hi"] = np.nanpercentile(bm[3], lo, axis=0), np.nanpercentile(bm[3], hi, axis=0)
        for row in mean_rows:
            c = ind_cols[row["method"]]
            for name, arr in (("mae", bm[0]), ("rmse", bm[1]), ("skill", bm[3])):
                dist = np.nanmean(arr[:, c], axis=1)
                row[f"{name}_lo"], row[f"{name}_hi"] = np.nanpercentile(dist, lo), np.nanpercentile(dist, hi)
    out = pd.concat([out, pd.DataFrame(mean_rows)], ignore_index=True)
    out["honest"] = ~out.method.eq("state_trend_oracle")
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    agg_order = {"OVERALL": 1, "MEAN": 2}
    out = out.assign(_m=out.method.map(order).fillna(99), _o=out.indicator_id.map(agg_order).fillna(0))
    out = out.sort_values(["_o", "indicator_id", "_m"]).drop(columns=["_m", "_o"])
    cols = ["indicator_id", "method", "honest", "n", "mae", "mae_lo", "mae_hi", "rmse", "rmse_lo",
            "rmse_hi", "skill", "skill_lo", "skill_hi", "noise_mse", "skill_ceiling"]
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def secondary_metrics(preds: pd.DataFrame, methods=None, bootstrap_reps: int = 1000, ci_level: float = 0.95,
                      seed: int = 0) -> pd.DataFrame:
    """CLAUDE.md section 5 metrics (DECISIONS D32), per indicator / OVERALL / MEAN and method:

    skill_vs_national (net of the national shift, with CI), skill_vs_rtm,
    noise_share_change = mean(se_prev^2 + se_true^2) / var(y_true - y_prev),
    ceiling_vs_persistence and ceiling_vs_rtm = 1 - noise_mse / MSE(baseline).
    """
    present = set(preds.method.unique())
    methods = [m for m in (methods or HONEST_METHODS) if m in present]
    need = sorted(set(methods) | {"persistence", "national_trend", "rtm"} & present)
    sub = preds[preds.method.isin(need)]
    vs_p = metric_table(sub, need, "persistence", 0, ci_level, seed)
    vs_n = metric_table(sub, need, "national_trend", bootstrap_reps, ci_level, seed)
    vs_r = metric_table(sub, need, "rtm", 0, ci_level, seed)
    key = ["indicator_id", "method"]
    out = vs_p[key + ["n", "skill", "skill_ceiling"]].rename(
        columns={"skill": "skill_vs_persistence", "skill_ceiling": "ceiling_vs_persistence"})
    out = out.merge(vs_n[key + ["skill", "skill_lo", "skill_hi"]].rename(
        columns={"skill": "skill_vs_national", "skill_lo": "skill_vs_national_lo",
                 "skill_hi": "skill_vs_national_hi"}), on=key)
    out = out.merge(vs_r[key + ["skill", "skill_ceiling"]].rename(
        columns={"skill": "skill_vs_rtm", "skill_ceiling": "ceiling_vs_rtm"}), on=key)
    # noise share of the variance of change (method-independent; persistence rows, common support)
    base = _common_support(sub, need)
    base = base[base.method == "persistence"].assign(chg=lambda d: d.y_true - d.y_prev,
                                                     noise=lambda d: d.se_prev ** 2 + d.se_true ** 2)
    per = base.groupby("indicator_id").agg(var_chg=("chg", "var"), noise=("noise", "mean"), n=("chg", "size"))
    share = (per.noise / per.var_chg).to_dict()
    share["OVERALL"] = float((per.noise * per.n).sum() / (per.var_chg * per.n).sum())
    share["MEAN"] = float(np.mean([v for k, v in share.items() if k != "OVERALL"]))
    out["noise_share_change"] = out.indicator_id.map(share)
    return out[out.method.isin(methods)].reset_index(drop=True)


# =====================================================================
# Paired tests (DECISIONS D32)
# =====================================================================
def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs are left as NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def paired_tests(preds: pd.DataFrame, comparisons=None, bootstrap_reps: int = 1000, ci_level: float = 0.95,
                 seed: int = 0, alpha: float = 0.05, cluster: str = "state") -> pd.DataFrame:
    """Paired cluster-bootstrap test of dMSE = MSE(method) - MSE(reference), per indicator and OVERALL.

    Both methods are evaluated on the same rows and the same bootstrap
    resamples of clusters (states). p = 2 * min(share of dMSE* <= 0, share
    >= 0) (percentile-interval inversion), BH-adjusted across indicators
    within each comparison. `beats_reference` requires BH p < alpha and dMSE < 0.
    """
    present = set(preds.method.unique())
    comps = [(a, b) for a, b in (comparisons or PAIRED_COMPARISONS) if a in present and b in present]
    rows = []
    lo, hi = 100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2
    for a, b in comps:
        keys, clusters, (N, _, S, _) = _cluster_matrices(preds, [a, b], cluster)
        W = _bootstrap_weights(len(clusters), bootstrap_reps, seed)
        part = []
        for ind in keys.indicator_id.unique():
            ia = keys.index[(keys.indicator_id == ind) & (keys.method == a)][0]
            ib = keys.index[(keys.indicator_id == ind) & (keys.method == b)][0]
            n = N[:, ia].sum()
            mse_a, mse_b = S[:, ia].sum() / n, S[:, ib].sum() / n
            nb = W @ N[:, ia]
            with np.errstate(invalid="ignore", divide="ignore"):
                d = (W @ S[:, ia] - W @ S[:, ib]) / nb
            p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
            part.append({"comparison": f"{a} vs {b}", "method": a, "reference": b, "indicator_id": ind,
                         "n": int(n), "mse_method": mse_a, "mse_reference": mse_b, "diff_mse": mse_a - mse_b,
                         "diff_lo": np.nanpercentile(d, lo), "diff_hi": np.nanpercentile(d, hi),
                         "skill_vs_reference": 1 - mse_a / mse_b, "p_boot": p})
        part = pd.DataFrame(part)
        is_ind = part.indicator_id != "OVERALL"
        part["p_bh"] = np.nan
        part.loc[is_ind, "p_bh"] = bh_adjust(part.loc[is_ind, "p_boot"])
        part.loc[~is_ind, "p_bh"] = part.loc[~is_ind, "p_boot"]
        part["beats_reference"] = (part.p_bh < alpha) & (part.diff_mse < 0)
        part["worse_than_reference"] = (part.p_bh < alpha) & (part.diff_mse > 0)
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.assign(_o=out.indicator_id.eq("OVERALL")).sort_values(["comparison", "_o", "indicator_id"])
    return out.drop(columns="_o").reset_index(drop=True)


def paired_summary(tests: pd.DataFrame) -> pd.DataFrame:
    """One row per comparison: indicators where the method is significantly better / worse."""
    t = tests[tests.indicator_id != "OVERALL"]
    o = tests[tests.indicator_id == "OVERALL"].set_index("comparison")
    s = t.groupby("comparison").agg(indicators=("indicator_id", "size"),
                                    better_raw=("diff_mse", lambda d: int((d < 0).sum())),
                                    significantly_better=("beats_reference", "sum"),
                                    significantly_worse=("worse_than_reference", "sum"))
    s["overall_diff_mse"] = o.diff_mse
    s["overall_diff_lo"], s["overall_diff_hi"], s["overall_p"] = o.diff_lo, o.diff_hi, o.p_boot
    return s.reset_index()


def deff_sensitivity(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fay-Herriot skill and noise ceiling under the deff sensitivity range."""
    deff0 = cfg["noise"]["deff"]
    rows = []
    for deff in cfg["noise"]["deff_sensitivity"]:
        name = "fay_herriot" if deff == deff0 else f"fay_herriot_deff{deff:g}"
        sub = preds[preds.method.isin(["persistence", name])].copy()
        sub["se_true"] = sub.se_true * np.sqrt(deff / deff0)
        sub["method"] = sub.method.replace({name: "fay_herriot"})
        t = metric_table(sub, ["persistence", "fay_herriot"], bootstrap_reps=0)
        t = t[t.method == "fay_herriot"].assign(deff=deff)
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
    methods = [m for m in PLOT_METHODS if m in set(results.method)]
    r = results[results.method.isin(methods)].copy()
    best_ml = (r[r.method.isin(["elastic_net", "lightgbm"]) & ~r.indicator_id.isin(["OVERALL", "MEAN"])]
               .groupby("indicator_id").skill.max().sort_values(ascending=False))
    order = ["OVERALL", "MEAN"] + list(best_ml.index)
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
