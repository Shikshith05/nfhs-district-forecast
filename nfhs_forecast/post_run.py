"""Post-run analyses on a predictions table in the NFHS-5 -> NFHS-6 format (DECISIONS D34, D37).

Everything here runs AFTER the one-shot evaluation, on its predictions file
(outputs/predictions_nfhs5to6.parquet), or in DRYRUN mode on the NFHS-4 -> NFHS-5
LOSO predictions as a stand-in. Nothing is refitted: the frozen models are
not touched; NFHS-6 values are used only to score and analyse forecasts.

Pre-registered (D32, D34): results on the harmonised set and the stable subset,
paired tests, secondary metrics, conformal coverage (split / normalised /
weighted), BH-FDR surprises vs naive top-k, per-district forecastability.
Added after the pre-registration and labelled exploratory (D37): rows by unit
group (flagged rows separately), the NFHS-4->5 vs NFHS-5->6 comparison and the
structural-break check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import conformal, evaluate, forecastability, surprise
from .config import region_of, write_csv, write_text

DRYRUN = "DRYRUN"
DRYRUN_BANNER = ("> **DRYRUN — NOT NFHS-6 RESULTS.** Produced from the NFHS-4 → NFHS-5 leave-one-state-out "
                 "predictions as a stand-in, only to test the post-run pipeline. Never cite these numbers.")
PRIMARY_CANDIDATES = ["national_trend", "state_trend", "rtm", "fay_herriot", "fh_rtm", "elastic_net", "lightgbm"]
GROUP_LABELS = {"stable": "stable-boundary districts", "changed_4to5": "identity, boundary changed NFHS-4→5",
                "flagged": "flagged: inherited / aggregated units"}
INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e1dfda", "#2a78d6", "#eb6834"


# =====================================================================
# Output writer (marks DRYRUN files so they cannot pass for real results)
# =====================================================================
@dataclass
class Out:
    root: Path
    dryrun: bool = False

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []

    def path(self, name: str) -> Path:
        return self.root / (f"{DRYRUN}_{name}" if self.dryrun else name)

    def _log(self, p: Path) -> Path:
        self.written.append(p.name)
        return p

    def csv(self, name: str, df: pd.DataFrame) -> Path:
        df = df.copy()
        df.insert(0, "run_mode", DRYRUN if self.dryrun else "NFHS6")
        p = self.path(name)
        write_csv(df, p, float_format="%.6g")
        return self._log(p)

    def json(self, name: str, obj) -> Path:
        obj = {"run_mode": DRYRUN if self.dryrun else "NFHS6", **obj}
        p = self.path(name)
        write_text(p, json.dumps(obj, indent=1, default=_json_default) + "\n")
        return self._log(p)

    def md(self, name: str, text: str) -> Path:
        if self.dryrun:
            text = DRYRUN_BANNER + "\n\n" + text
        p = self.path(name)
        write_text(p, text)
        return self._log(p)

    def png(self, name: str) -> Path:
        """Path for a figure; call watermark() after saving it."""
        return self._log(self.path(name))

    def watermark(self, p: Path) -> None:
        if self.dryrun:
            watermark_png(p, "DRYRUN — NOT NFHS-6 RESULTS")


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def watermark_png(path: Path, text: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = plt.imread(path)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=max(12, w / 30), color="#c0392b", alpha=0.35,
            ha="center", va="center", rotation=25, weight="bold")
    ax.text(0.99, 0.01, text, transform=ax.transAxes, fontsize=12, color="#c0392b", ha="right", va="bottom",
            weight="bold")
    fig.savefig(path, dpi=100)
    plt.close(fig)


# =====================================================================
# Inputs
# =====================================================================
def choose_primary(results_45: pd.DataFrame, rule: str = "auto") -> str:
    """Pre-registered rule (D34): lowest pooled LOSO RMSE on NFHS-4 -> NFHS-5 among the primary methods."""
    if rule != "auto":
        return rule
    r = results_45[(results_45.indicator_id == "OVERALL") & results_45.method.isin(PRIMARY_CANDIDATES)]
    return str(r.sort_values(["rmse", "method"]).iloc[0].method)


def unit_group(unit_type: pd.Series, stable: pd.Series) -> pd.Series:
    flagged = unit_type.isin(["inherited_remnant", "inherited_child", "aggregated", "identity_renamed"])
    return pd.Series(np.select([stable.astype(bool), flagged], ["stable", "flagged"], "changed_4to5"),
                     index=unit_type.index)


def prepare_target(target: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Annotate unit_type / stable_unit / parent_uid if missing; add `group`; keep honest methods."""
    t = target[target.method.isin(evaluate.HONEST_METHODS)].copy()
    u = units.set_index("unit_uid")
    if "unit_type" not in t.columns:
        t["unit_type"] = t.district_uid.map(u.unit_type).fillna("identity")
    if "parent_uid" not in t.columns:
        t["parent_uid"] = t.district_uid.map(u.parent_uid).fillna(t.district_uid)
    if "stable_unit" not in t.columns:
        t["stable_unit"] = t.district_uid.map(u.stable_unit).fillna(False).astype(bool)
    t["stable_unit"] = t.stable_unit.astype(bool)
    t["group"] = unit_group(t.unit_type, t.stable_unit)
    return t


def dryrun_standin(preds_45: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """NFHS-4 -> NFHS-5 LOSO predictions reshaped into the NFHS-5 -> NFHS-6 predictions format."""
    t = preds_45[preds_45.method.isin(evaluate.HONEST_METHODS)].copy()
    t = prepare_target(t.drop(columns=[c for c in ("unit_type", "parent_uid", "stable_unit") if c in t]), units)
    t["run_mode"] = DRYRUN
    return t.drop(columns="group")


# =====================================================================
# Analyses
# =====================================================================
def results_by_subset(t: pd.DataFrame, kw: dict) -> dict[str, pd.DataFrame]:
    subsets = {"all": t, "stable": t[t.stable_unit]}
    for g in GROUP_LABELS:
        if g != "stable":
            subsets[g] = t[t.group == g]
    out = {}
    for name, sub in subsets.items():
        if sub.empty:
            continue
        res = evaluate.metric_table(sub, evaluate.HONEST_METHODS, **(kw if name in ("all", "stable") else
                                                                     {**kw, "bootstrap_reps": 0}))
        res.insert(0, "subset", name)
        res.insert(1, "n_districts", sub.district_uid.nunique())
        out[name] = res
    return out


def comparison(res45: pd.DataFrame, res56: pd.DataFrame, primary: str) -> tuple[pd.DataFrame, dict]:
    """Skill per method x indicator in NFHS-4->5 LOSO CV vs NFHS-5->6 (exploratory, D37)."""
    key = ["indicator_id", "method"]
    a = res45[key + ["skill", "rmse"]].rename(columns={"skill": "skill_45", "rmse": "rmse_45"})
    b = res56[key + ["skill", "rmse", "skill_lo", "skill_hi"]].rename(
        columns={"skill": "skill_56", "rmse": "rmse_56", "skill_lo": "skill_56_lo", "skill_hi": "skill_56_hi"})
    m = a.merge(b, on=key)
    m["skill_change"] = m.skill_56 - m.skill_45
    over = m[m.indicator_id == "OVERALL"].set_index("method")
    rank45, rank56 = over.skill_45.rank(ascending=False), over.skill_56.rank(ascending=False)
    per = m[(m.method == primary) & ~m.indicator_id.isin(["OVERALL", "MEAN"])]
    summ = {
        "primary_method": primary,
        "methods_compared": int(len(over)),
        "spearman_methods_overall_skill": float(rank45.corr(rank56, method="spearman")) if len(over) > 2 else np.nan,
        "best_method_45": str(over.skill_45.idxmax()), "best_method_56": str(over.skill_56.idxmax()),
        "spearman_indicators_primary": float(per.skill_45.corr(per.skill_56, method="spearman")),
        "primary_overall_skill_45": float(over.loc[primary, "skill_45"]),
        "primary_overall_skill_56": float(over.loc[primary, "skill_56"]),
    }
    return m.sort_values(key).reset_index(drop=True), summ


def _ols_from_sums(n, sx, sy, sxx, sxy):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sxy - sx * sy / n) / (sxx - sx ** 2 / n)


def structural_break(cal: pd.DataFrame, tgt: pd.DataFrame, primary: str, reps: int = 1000, seed: int = 0,
                     q: float = 0.05) -> pd.DataFrame:
    """Exploratory (D37): did the NFHS-4->5 relation hold in NFHS-5->6? Per indicator:

    shift   mean change y_t+1 - y_t (the national shift) in each period
    slope   OLS slope of y_t+1 on y_t (regression to the mean) in each period
    scale   RMS error of the primary method in each period
    Differences (5->6 minus 4->5) get 95% CIs from a state-clustered bootstrap
    run independently in each period; p-values are BH-adjusted over indicators
    per statistic. `break_any` = any statistic significant at q.
    """
    rng = np.random.default_rng(seed)

    def sums(df):
        base = df[df.method == "persistence"].dropna(subset=["y_true", "y_prev"])
        res = df[df.method == primary].dropna(subset=["y_true", "y_pred"])
        b = base.assign(x=base.y_prev, y=base.y_true, d=base.y_true - base.y_prev)
        g = b.assign(xx=b.x ** 2, xy=b.x * b.y).groupby(["indicator_id", "state"]).agg(
            n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxx=("xx", "sum"), sxy=("xy", "sum"), sd=("d", "sum"))
        r = res.assign(e2=(res.y_true - res.y_pred) ** 2).groupby(["indicator_id", "state"]).agg(
            ne=("e2", "size"), se2=("e2", "sum"))
        return g.join(r, how="outer").fillna(0.0)

    S45, S56 = sums(cal), sums(tgt)
    rows = []
    for ind in sorted(set(S45.index.get_level_values(0)) & set(S56.index.get_level_values(0))):
        stats = {}
        for lab, S in (("45", S45.loc[ind]), ("56", S56.loc[ind])):
            M = S.to_numpy(float)
            W = rng.multinomial(len(S), np.full(len(S), 1 / len(S)), size=reps).astype(float)
            full = M.sum(axis=0)[None, :]
            for name, arr in (("point", full), ("boot", W @ M)):
                n, sx, sy, sxx, sxy, sd, ne, se2 = arr.T
                with np.errstate(invalid="ignore", divide="ignore"):
                    stats[(lab, name)] = {"shift": sd / n, "slope": _ols_from_sums(n, sx, sy, sxx, sxy),
                                          "scale": np.sqrt(se2 / ne)}
        row = {"indicator_id": ind}
        for s in ("shift", "slope", "scale"):
            p45, p56 = stats[("45", "point")][s][0], stats[("56", "point")][s][0]
            d = stats[("56", "boot")][s] - stats[("45", "boot")][s]
            d = d[np.isfinite(d)]
            row.update({f"{s}_45": p45, f"{s}_56": p56, f"{s}_diff": p56 - p45,
                        f"{s}_diff_lo": np.percentile(d, 2.5) if len(d) else np.nan,
                        f"{s}_diff_hi": np.percentile(d, 97.5) if len(d) else np.nan,
                        f"{s}_p": min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))) if len(d) else np.nan})
        rows.append(row)
    out = pd.DataFrame(rows)
    for s in ("shift", "slope", "scale"):
        out[f"{s}_p_bh"] = evaluate.bh_adjust(out[f"{s}_p"])
        out[f"{s}_break"] = out[f"{s}_p_bh"] < q
    out["break_any"] = out[[f"{s}_break" for s in ("shift", "slope", "scale")]].any(axis=1)
    out["primary_method"] = primary
    return out


# =====================================================================
# Figures
# =====================================================================
def _style(ax):
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7.5)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_comparison(comp: pd.DataFrame, primary: str, path: Path, label56: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=150)
    ind = comp[~comp.indicator_id.isin(["OVERALL", "MEAN"])]
    lim = [min(-0.2, np.nanmin(ind[["skill_45", "skill_56"]].to_numpy()) - 0.05), 1.0]
    ax.plot(lim, lim, color=INK2, lw=0.9, ls="--", zorder=1)
    for m, color, marker in (("rtm", ORANGE, "s"), (primary, BLUE, "o")):
        d = ind[ind.method == m]
        ax.scatter(d.skill_45, d.skill_56, s=34, color=color, marker=marker, edgecolors="white", linewidths=0.8,
                   zorder=3, label=evaluate.METHOD_LABELS.get(m, m))
    d = ind[ind.method == primary]
    for r in d.assign(gap=(d.skill_56 - d.skill_45).abs()).nlargest(4, "gap").itertuples():
        ax.annotate(r.indicator_id, (r.skill_45, r.skill_56), fontsize=6.8, color=INK2, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Skill, NFHS-4 → NFHS-5 (leave-one-state-out CV)", fontsize=8, color=INK)
    ax.set_ylabel(f"Skill, {label56}", fontsize=8, color=INK)
    ax.set_title("Did per-indicator skill hold out of time?", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _style(ax)
    fig.text(0.01, 0.005, "Each point is one indicator. Dashed line: equal skill in both periods.", fontsize=6.8,
             color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_coverage(cov: pd.DataFrame, nominal: float, path: Path) -> None:
    plt = _plt()
    d = cov[(cov.indicator_id == "OVERALL") & (cov.get("subset", "all") == "all")].copy()
    d["label"] = d.variant + " · " + d.method.map(lambda m: evaluate.METHOD_LABELS.get(m, m))
    d = d.sort_values(["variant", "method"])
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(d) + 1.4), dpi=150)
    y = np.arange(len(d))
    half = 1.96 * d.coverage_se.to_numpy()
    ax.axvline(nominal, color=INK2, lw=0.9, ls="--", zorder=1)
    ax.hlines(y, d.coverage - half, d.coverage + half, color=BLUE, lw=1.2, alpha=0.6, zorder=2)
    ax.scatter(d.coverage, y, s=30, color=BLUE, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label, fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Empirical coverage (95% binomial interval)", fontsize=8, color=INK)
    ax.set_title(f"Conformal interval coverage; dashed line = nominal {nominal:.0%}", fontsize=9.5, color=INK,
                 loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_surprises(tab: pd.DataFrame, path: Path, method: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    base = tab[~tab.surprise & ~tab.naive_topk]
    ax.scatter(base.se_true, base.resid, s=5, color="#c9c7c1", zorder=1, label="other district × indicator")
    nv = tab[tab.naive_topk & ~tab.surprise]
    ax.scatter(nv.se_true, nv.resid, s=26, facecolors="none", edgecolors=ORANGE, marker="s", linewidths=1.1,
               zorder=3, label="naive top-k only")
    bh = tab[tab.surprise]
    ax.scatter(bh.se_true, bh.resid, s=26, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4,
               label="BH-FDR surprise")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Sampling SE of the observed value (assumed)", fontsize=8, color=INK)
    ax.set_ylabel("Observed − forecast (percentage points)", fontsize=8, color=INK)
    ax.set_title(f"Surprises ({evaluate.METHOD_LABELS.get(method, method)}): FDR control vs naive top-k",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_break(br: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    d = br.sort_values("indicator_id")
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 0.27 * len(d) + 1.6), dpi=150, sharey=True)
    titles = {"shift": "Mean change (pp)", "slope": "RTM slope", "scale": "RMS error, primary (pp)"}
    for ax, s in zip(axes, ("shift", "slope", "scale")):
        sig = d[f"{s}_break"].to_numpy()
        ax.axvline(0, color=INK2, lw=0.9)
        ax.hlines(y, d[f"{s}_diff_lo"], d[f"{s}_diff_hi"], color=BLUE, lw=1.1, alpha=0.6)
        ax.scatter(d[f"{s}_diff"][sig], y[sig], s=22, color=BLUE, edgecolors="white", linewidths=0.8, zorder=3)
        ax.scatter(d[f"{s}_diff"][~sig], y[~sig], s=22, facecolors="white", edgecolors=BLUE, linewidths=1.0, zorder=3)
        ax.set_title(f"Δ {titles[s]}", fontsize=8.5, color=INK, loc="left")
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d.indicator_id, fontsize=7, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Structural-break check: NFHS-5→6 minus NFHS-4→5 (exploratory)", fontsize=9.5, color=INK, x=0.01,
                 ha="left")
    fig.text(0.01, 0.005, "Lines: 95% CI, state-clustered bootstrap per period. Filled: BH-adjusted p < 0.05.",
             fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# =====================================================================
# Driver
# =====================================================================
def run(target: pd.DataFrame, cal: pd.DataFrame, res45: pd.DataFrame, units: pd.DataFrame, cfg: dict,
        out: Out, codes: pd.Series | None = None, label56: str = "NFHS-5 → NFHS-6 (frozen models)") -> dict:
    """All post-run tables, figures and the key-number summary. Returns the summary dict."""
    g, pr = cfg["gate_c"], cfg.get("post_run", {})
    ev = cfg["evaluation"]
    reps = int(pr.get("bootstrap_reps", ev["bootstrap_reps"]))
    kw = dict(bootstrap_reps=reps, ci_level=ev["ci_level"], seed=cfg["random_seed"])
    t = prepare_target(target, units)
    primary = choose_primary(res45, g["primary_method"])
    alpha = g["alpha"]

    # results by subset (pre-registered: all, stable; exploratory: groups)
    res = results_by_subset(t, kw)
    for name, r in res.items():
        out.csv(f"results_{name}.csv", r)
    out.csv("results_subsets.csv", pd.concat(res.values(), ignore_index=True))
    for name, sub in (("all", t), ("stable", t[t.stable_unit])):
        tests = evaluate.paired_tests(sub, **kw)
        out.csv(f"paired_tests_{name}.csv", tests)
        out.csv(f"paired_summary_{name}.csv", evaluate.paired_summary(tests))
        out.csv(f"secondary_{name}.csv", evaluate.secondary_metrics(sub, **kw))
    counts = (t[t.method == "persistence"].dropna(subset=["y_true"]).groupby("group")
              .agg(districts=("district_uid", "nunique"), pairs=("district_uid", "size")).reset_index())
    out.csv("group_counts.csv", counts)

    # comparison with NFHS-4 -> NFHS-5 (exploratory)
    comp, comp_sum = comparison(res45, res["all"], primary)
    out.csv("comparison_45_vs_56.csv", comp)

    # C1 conformal
    iv = conformal.conformal_intervals(cal, t, alpha, methods=sorted(set(g["conformal_methods"]) | {primary}),
                                       region=region_of(cfg), weight_clip=g["weight_clip"], seed=cfg["random_seed"])
    cov = pd.concat([conformal.coverage_table(iv, alpha).assign(subset="all"),
                     conformal.coverage_table(iv[iv.stable_unit.astype(bool)], alpha).assign(subset="stable")],
                    ignore_index=True)
    out.csv("coverage.csv", cov)

    # structural break (exploratory)
    br = structural_break(cal, t, primary, reps=int(pr.get("break_bootstrap_reps", 1000)), seed=cfg["random_seed"],
                          q=float(pr.get("break_q", 0.05)))
    out.csv("structural_break.csv", br)

    # C2 surprises
    surp = {}
    for m in sorted({primary, "rtm"}):
        tab, s = surprise.detect_surprises(cal, t, m, q=g["fdr_q"], floor_share=g["sigma_floor_share"],
                                           naive_k_min=g["naive_k_min"])
        out.csv(f"surprises_{m}.csv", tab)
        surp[m] = (tab, s)
    out.csv("surprise_summary.csv", pd.DataFrame([s for _, s in surp.values()]))

    # C3 forecastability
    dist = forecastability.district_table(t, primary)
    indt = forecastability.indicator_table(t, primary)
    dec = forecastability.decompose(t, primary)
    out.csv("forecastability_districts.csv", dist)
    out.csv("forecastability_indicators.csv", indt)
    out.csv("forecastability_decomposition.csv", pd.DataFrame([dec]))

    # figures
    for name, sub in (("all", res["all"]), ("stable", res.get("stable"))):
        if sub is not None:
            p = out.png(f"skill_by_indicator_{name}.png")
            evaluate.plot_skill(sub, p, f"Skill vs persistence, {label56} ({name})")
            out.watermark(p)
    for fn, name, args in ((fig_comparison, "comparison_45_vs_56.png", (comp, primary)),
                           (fig_coverage, "coverage.png", (cov, 1 - alpha)),
                           (fig_surprises, "surprises.png", (surp[primary][0],)),
                           (fig_break, "structural_break.png", (br,))):
        p = out.png(name)
        if fn is fig_comparison:
            fn(*args, p, label56)
        elif fn is fig_surprises:
            fn(args[0], p, primary)
        else:
            fn(*args, p)
        out.watermark(p)
    n_mapped = 0
    if codes is not None and len(codes):
        s = dist.set_index("district_uid")
        for col, name, lab in (("skill", "map_skill.png", "1 − mean normalised squared error"),
                               ("ceiling", "map_ceiling.png", "1 − mean noise share")):
            p = out.png(name)
            n_mapped = forecastability.plot_map(s[col], codes, cfg["abs_paths"]["geojson"], p,
                                                f"{'Forecast skill' if col == 'skill' else 'Noise ceiling'} by "
                                                f"district ({primary})", lab)
            out.watermark(p)

    cov_all = cov[(cov.subset == "all") & (cov.indicator_id == "OVERALL")]
    ps = surp[primary][1]
    summary = {
        "primary_method": primary, "alpha": alpha, "fdr_q": g["fdr_q"],
        "n_districts": int(t.district_uid.nunique()), "n_stable": int(t.loc[t.stable_unit, "district_uid"].nunique()),
        "n_flagged": int(t.loc[t.group == "flagged", "district_uid"].nunique()),
        "n_changed_4to5": int(t.loc[t.group == "changed_4to5", "district_uid"].nunique()),
        "coverage_split_primary": float(cov_all[(cov_all.variant == "split") & (cov_all.method == primary)].coverage.iloc[0]),
        "coverage_weighted_primary": float(cov_all[(cov_all.variant == "weighted") & (cov_all.method == primary)].coverage.iloc[0]),
        "coverage_normalised_primary": float(cov_all[(cov_all.variant == "normalised") & (cov_all.method == primary)].coverage.iloc[0]),
        "surprises_primary": int(ps["surprises_normal"]), "surprise_tests_primary": int(ps["tests"]),
        "surprise_overlap_naive": int(ps["overlap_naive_vs_bh"]), "naive_k": int(ps["naive_k"]),
        "naive_mean_se": ps["naive_topk_mean_se_true"], "bh_mean_se": ps["bh_mean_se_true"],
        "breaks_any": int(br.break_any.sum()), "breaks_shift": int(br.shift_break.sum()),
        "breaks_slope": int(br.slope_break.sum()), "breaks_scale": int(br.scale_break.sum()),
        "decomp_r2_indicator": dec["r2_indicator_only"], "decomp_r2_district": dec["r2_district_only"],
        "decomp_r2_additive": dec["r2_additive"], "decomp_unique_indicator": dec["unique_indicator"],
        "decomp_unique_district": dec["unique_district"], "n_mapped": int(n_mapped),
        **{f"cmp_{k}": v for k, v in comp_sum.items()},
    }
    out.csv("key_numbers.csv", pd.DataFrame([{"key": k, "value": v} for k, v in summary.items()]))
    out.json("post_run_summary.json", {"summary": summary, "files": sorted(set(out.written))})
    return summary
