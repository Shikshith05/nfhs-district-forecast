"""Render docs/report_template.md into the report, with every result number pulled from a file (D38).

Every result in the template is a slot `{{name}}`. SLOTS below says where its
value comes from (file, row filter, column, format). Rendered values are
wrapped in invisible markers `<!--slot:name-->value<!--/slot-->` so that
scripts/check_report_numbers.py can re-derive and compare each one.

Sources:  out:<path>    a file under outputs/ (NFHS-4 -> NFHS-5 results, harmonisation, manifest)
          run:<name>    a post-run file (NFHS-5 -> NFHS-6): outputs/post_run/<name> in final
                        mode, outputs/dryrun/DRYRUN_<name> in dryrun mode, PENDING in prerun mode
Modes:    prerun   docs/report.md now: NFHS-6 slots shown as [[NFHS-6 PENDING]]
          final    docs/report.md after Gate B: everything filled from outputs/post_run/
          dryrun   outputs/dryrun/DRYRUN_report.md from the DRYRUN stand-in files
`{primary}` in a source or filter is the pre-registered primary method (D34),
chosen from NFHS-4 -> NFHS-5 results only.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_text

PENDING = "[[NFHS-6 PENDING]]"
SLOT_RE = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")
MARK_RE = re.compile(r"<!--slot:([a-zA-Z0-9_]+)-->(.*?)<!--/slot-->", re.S)
METHOD_LABELS = {
    "persistence": "Persistence", "national_trend": "National trend", "state_trend": "State trend",
    "rtm": "Regression to the mean (RTM)", "fay_herriot": "Fay-Herriot", "fh_rtm": "FH + RTM",
    "elastic_net": "Elastic net", "lightgbm": "LightGBM", "elastic_net__own": "Elastic net, own history",
    "lightgbm__own": "LightGBM, own history", "elastic_net__cross": "Elastic net, + other indicators",
    "lightgbm__cross": "LightGBM, + other indicators", "elastic_net__cross_spatial": "Elastic net, + spatial",
    "lightgbm__cross_spatial": "LightGBM, + spatial", "elastic_net__nostate": "Elastic net, no state dummies",
    "lightgbm__nostate": "LightGBM, no state dummies", "state_trend_oracle": "State trend (oracle)",
    "elastic_net vs rtm": "Elastic net vs RTM", "lightgbm vs rtm": "LightGBM vs RTM",
    "elastic_net__nostate vs rtm": "Elastic net (no state dummies) vs RTM",
    "lightgbm__nostate vs rtm": "LightGBM (no state dummies) vs RTM",
    "all": "harmonised set (all)", "stable": "stable-boundary subset",
    "changed_4to5": "identity, boundary changed NFHS-4→5", "flagged": "flagged (inherited / aggregated)",
    "split": "split", "normalised": "normalised", "weighted": "shift-weighted",
}
F3, F2, F1, PCT = "{:.3f}", "{:.2f}", "{:.1f}", "{:.1%}"
MAIN_METHODS = ["persistence", "national_trend", "state_trend", "rtm", "fay_herriot", "fh_rtm", "elastic_net",
                "lightgbm", "elastic_net__nostate", "lightgbm__nostate"]
ABLATIONS = ["elastic_net__own", "elastic_net__cross", "elastic_net__cross_spatial", "elastic_net",
             "elastic_net__nostate", "lightgbm__own", "lightgbm__cross", "lightgbm__cross_spatial", "lightgbm",
             "lightgbm__nostate"]
COMPARISONS = ["elastic_net vs rtm", "lightgbm vs rtm", "elastic_net__nostate vs rtm", "lightgbm__nostate vs rtm"]


# ---------------------------------------------------------------------
# Slot specifications
# ---------------------------------------------------------------------
def S(src, col, fmt=F3, where=None, agg="value"):
    return {"kind": "scalar", "src": src, "col": col, "fmt": fmt, "where": where or {}, "agg": agg}


def K(key, fmt=F3):
    """A number from the post-run key_numbers.csv."""
    return S("run:key_numbers.csv", "value", fmt, {"key": key})


def T(src, row_col, rows, cols, where=None, label="", sort=None, head=None):
    """Table: one row per value of row_col (rows=None: all matching rows, optionally sorted / truncated).
    cols: list of (header, column, fmt[, extra where]); column 'ci:x' renders [x_lo, x_hi]."""
    return {"kind": "table", "src": src, "row_col": row_col, "rows": rows, "cols": cols, "where": where or {},
            "label": label, "sort": sort, "head": head}


def FIG(src, caption):
    return {"kind": "fig", "src": src, "caption": caption}


def V(src, where, lo="diff_lo", hi="diff_hi", neg="significantly lower MSE than", pos="significantly higher MSE than",
      zero="no significant difference in MSE from"):
    return {"kind": "verdict", "src": src, "where": where, "lo": lo, "hi": hi, "neg": neg, "pos": pos, "zero": zero}


def _skill_cols(src_all, src_stable=None):
    cols = [("Pooled skill", "skill", F3, {"indicator_id": "OVERALL"}),
            ("95% CI", "ci:skill", F3, {"indicator_id": "OVERALL"}),
            ("Mean skill over indicators", "skill", F3, {"indicator_id": "MEAN"})]
    if src_stable:
        cols.append(("Pooled skill, stable subset", "skill", F3, {"indicator_id": "OVERALL", "@src": src_stable}))
    return cols


R45, R45S, R45SEEN = "out:results_nfhs4to5.csv", "out:results_nfhs4to5_stable.csv", "out:results_nfhs4to5_seenstate.csv"
P45, SEC45, MAN = "out:paired_tests_nfhs4to5.csv", "out:secondary_nfhs4to5.csv", "out:models/manifest.json"
UNITS, REVIEW = "out:harmonised_units.csv", "out:crosswalk_review.csv"

SLOTS: dict[str, dict] = {
    # ---------------- design / data / harmonisation (pre-run facts) ----------------
    "n_train": S(MAN, "n_core_districts", "int"),
    "n_test": S(MAN, "n_test_units", "int"),
    "n_stable": S(MAN, "n_stable_units", "int"),
    "n_ind": S(MAN, "indicators", "int", agg="len"),
    "n_methods": S(MAN, "methods", "int", agg="len"),
    "train_hash": S(MAN, "train_hash", "text"),
    "config_hash": S(MAN, "config_hash", "text"),
    "n_test_changed45": S(UNITS, "unit_uid", "int", {"unit_type": "identity", "stable_unit": "False",
                                                     "test_unit": "True"}, agg="count"),
    "n_test_inherited": S(UNITS, "unit_uid", "int", {"unit_type": "inherited_remnant", "test_unit": "True"}, agg="count"),
    "n_train_only": S(UNITS, "unit_uid", "int", {"unit_type": "identity_train_only"}, agg="count"),
    "n_train_added": S(UNITS, "unit_uid", "int", {"train_unit": "True", "stable_unit": "False"}, agg="count"),
    "n_review": S(REVIEW, "relation", "int", agg="count"),
    "n_rev_45": S(REVIEW, "relation", "int", {"relation": "boundary_changed_4to5_parent_unknown"}, agg="count"),
    "n_rev_new": S(REVIEW, "relation", "int", {"relation": "new_district_parent_unknown"}, agg="count"),
    "n_rev_retained": S(REVIEW, "relation", "int", {"relation": "retained_name_boundary_changed"}, agg="count"),
    "n_rev_low": S(REVIEW, "relation", "int", {"relation": "split_candidate_single_flagged_parent"}, agg="count"),
    # ---------------- NFHS-4 -> NFHS-5 results ----------------
    "primary_method": {"kind": "primary"},
    "s45_rtm": S(R45, "skill", F3, {"indicator_id": "OVERALL", "method": "rtm"}),
    "s45_nat": S(R45, "skill", F3, {"indicator_id": "OVERALL", "method": "national_trend"}),
    "s45_en": S(R45, "skill", F3, {"indicator_id": "OVERALL", "method": "elastic_net"}),
    "s45_lgbm": S(R45, "skill", F3, {"indicator_id": "OVERALL", "method": "lightgbm"}),
    "s45_fh": S(R45, "skill", F3, {"indicator_id": "OVERALL", "method": "fay_herriot"}),
    "s45_oracle": S(R45, "skill", F3, {"indicator_id": "OVERALL", "method": "state_trend_oracle"}),
    "s45_rtm_mean": S(R45, "skill", F3, {"indicator_id": "MEAN", "method": "rtm"}),
    "s45_en_mean": S(R45, "skill", F3, {"indicator_id": "MEAN", "method": "elastic_net"}),
    "s45_en_stable": S(R45S, "skill", F3, {"indicator_id": "OVERALL", "method": "elastic_net"}),
    "s45_rtm_stable": S(R45S, "skill", F3, {"indicator_id": "OVERALL", "method": "rtm"}),
    "t45_main": T(R45, "method", MAIN_METHODS + ["state_trend_oracle"], _skill_cols(R45, R45S), label="Method"),
    "d45_en_rtm": S(P45, "diff_mse", F2, {"comparison": "elastic_net vs rtm", "indicator_id": "OVERALL"}),
    "d45_en_rtm_lo": S(P45, "diff_lo", F2, {"comparison": "elastic_net vs rtm", "indicator_id": "OVERALL"}),
    "d45_en_rtm_hi": S(P45, "diff_hi", F2, {"comparison": "elastic_net vs rtm", "indicator_id": "OVERALL"}),
    "v45_en_rtm": V(P45, {"comparison": "elastic_net vs rtm", "indicator_id": "OVERALL"}),
    "v45_lgbm_rtm": V(P45, {"comparison": "lightgbm vs rtm", "indicator_id": "OVERALL"}),
    "n45_en_better": S("out:paired_summary_nfhs4to5.csv", "significantly_better", "int", {"comparison": "elastic_net vs rtm"}),
    "n45_en_better_stable": S("out:paired_summary_nfhs4to5_stable.csv", "significantly_better", "int",
                              {"comparison": "elastic_net vs rtm"}),
    "t45_paired": T("out:paired_summary_nfhs4to5.csv", "comparison", COMPARISONS,
                    [("Indicators better (raw)", "better_raw", "int"), ("Significantly better (BH)", "significantly_better", "int"),
                     ("Significantly worse (BH)", "significantly_worse", "int"), ("Pooled ΔMSE", "overall_diff_mse", F2),
                     ("95% CI", "ci:overall_diff", F2), ("p", "overall_p", F3)], label="Comparison"),
    "t45_ablation": T(R45, "method", ABLATIONS, [("Pooled skill", "skill", F3, {"indicator_id": "OVERALL"}),
                                                 ("Mean skill", "skill", F3, {"indicator_id": "MEAN"})], label="Feature set"),
    "noise_share_45": S(SEC45, "noise_share_change", PCT, {"indicator_id": "OVERALL", "method": "rtm"}),
    "noise_share_45_mean": S(SEC45, "noise_share_change", PCT, {"indicator_id": "MEAN", "method": "rtm"}),
    "ceil_p_45": S(SEC45, "ceiling_vs_persistence", F3, {"indicator_id": "OVERALL", "method": "rtm"}),
    "ceil_rtm_45": S(SEC45, "ceiling_vs_rtm", F3, {"indicator_id": "OVERALL", "method": "rtm"}),
    "t45_secondary": T(SEC45, "method", ["rtm", "fh_rtm", "elastic_net", "lightgbm"],
                       [("Skill vs persistence", "skill_vs_persistence", F3, {"indicator_id": "OVERALL"}),
                        ("Skill net of national shift", "skill_vs_national", F3, {"indicator_id": "OVERALL"}),
                        ("95% CI", "ci:skill_vs_national", F3, {"indicator_id": "OVERALL"}),
                        ("Skill vs RTM", "skill_vs_rtm", F3, {"indicator_id": "OVERALL"})], label="Method"),
    "s45_seen_en": S(R45SEEN, "skill", F3, {"indicator_id": "OVERALL", "method": "elastic_net"}),
    "s45_seen_rtm": S(R45SEEN, "skill", F3, {"indicator_id": "OVERALL", "method": "rtm"}),
    "s45_seen_st": S(R45SEEN, "skill", F3, {"indicator_id": "OVERALL", "method": "state_trend"}),
    "s45_seen_en_nostate": S(R45SEEN, "skill", F3, {"indicator_id": "OVERALL", "method": "elastic_net__nostate"}),
    "fh_deff15": S("out:sensitivity_deff.csv", "skill", F3, {"indicator_id": "OVERALL", "deff": "1.5"}),
    "fh_deff3": S("out:sensitivity_deff.csv", "skill", F3, {"indicator_id": "OVERALL", "deff": "3.0"}),
    "dev_cov_split": S("out:gate_c_dev/coverage.csv", "coverage", F3, {"indicator_id": "OVERALL", "variant": "split",
                                                                      "method": "elastic_net"}),
    "dev_cov_weighted": S("out:gate_c_dev/coverage.csv", "coverage", F3, {"indicator_id": "OVERALL",
                                                                         "variant": "weighted", "method": "elastic_net"}),
    "fig45_skill": FIG("out:skill_by_indicator.png", "Skill vs persistence by indicator, NFHS-4 → NFHS-5, leave-one-state-out CV"),
    # ---------------- NFHS-5 -> NFHS-6 (filled after Gate B) ----------------
    "run_status": {"kind": "status"},
    "n56": K("n_districts", "int"), "n56_stable": K("n_stable", "int"), "n56_flagged": K("n_flagged", "int"),
    "n56_changed": K("n_changed_4to5", "int"),
    "s56_rtm": S("run:results_all.csv", "skill", F3, {"indicator_id": "OVERALL", "method": "rtm"}),
    "s56_nat": S("run:results_all.csv", "skill", F3, {"indicator_id": "OVERALL", "method": "national_trend"}),
    "s56_st": S("run:results_all.csv", "skill", F3, {"indicator_id": "OVERALL", "method": "state_trend"}),
    "s56_primary": S("run:results_all.csv", "skill", F3, {"indicator_id": "OVERALL", "method": "{primary}"}),
    "s56_primary_lo": S("run:results_all.csv", "skill_lo", F3, {"indicator_id": "OVERALL", "method": "{primary}"}),
    "s56_primary_hi": S("run:results_all.csv", "skill_hi", F3, {"indicator_id": "OVERALL", "method": "{primary}"}),
    "s56_primary_stable": S("run:results_stable.csv", "skill", F3, {"indicator_id": "OVERALL", "method": "{primary}"}),
    "t56_main": T("run:results_all.csv", "method", MAIN_METHODS, _skill_cols("run:results_all.csv", "run:results_stable.csv"),
                  label="Method"),
    "t56_groups": T("run:results_subsets.csv", "subset", ["all", "stable", "changed_4to5", "flagged"],
                    [("Districts", "n_districts", "int", {"method": "rtm", "indicator_id": "OVERALL"}),
                     ("RTM skill", "skill", F3, {"method": "rtm", "indicator_id": "OVERALL"}),
                     ("Primary-method skill", "skill", F3, {"method": "{primary}", "indicator_id": "OVERALL"}),
                     ("Mean skill, primary", "skill", F3, {"method": "{primary}", "indicator_id": "MEAN"})],
                    label="Subset"),
    "t56_paired": T("run:paired_summary_all.csv", "comparison", COMPARISONS,
                    [("Significantly better (BH)", "significantly_better", "int"),
                     ("Significantly worse (BH)", "significantly_worse", "int"), ("Pooled ΔMSE", "overall_diff_mse", F2),
                     ("95% CI", "ci:overall_diff", F2), ("p", "overall_p", F3)], label="Comparison"),
    "v56_en_rtm": V("run:paired_tests_all.csv", {"comparison": "elastic_net vs rtm", "indicator_id": "OVERALL"}),
    "v56_lgbm_rtm": V("run:paired_tests_all.csv", {"comparison": "lightgbm vs rtm", "indicator_id": "OVERALL"}),
    "v56_en_rtm_stable": V("run:paired_tests_stable.csv", {"comparison": "elastic_net vs rtm", "indicator_id": "OVERALL"}),
    "t56_secondary": T("run:secondary_all.csv", "method", ["national_trend", "state_trend", "rtm", "fh_rtm", "elastic_net", "lightgbm"],
                       [("Skill net of national shift", "skill_vs_national", F3, {"indicator_id": "OVERALL"}),
                        ("95% CI", "ci:skill_vs_national", F3, {"indicator_id": "OVERALL"}),
                        ("Skill vs RTM", "skill_vs_rtm", F3, {"indicator_id": "OVERALL"})], label="Method"),
    "noise_share_56": S("run:secondary_all.csv", "noise_share_change", PCT, {"indicator_id": "OVERALL", "method": "rtm"}),
    "ceil_rtm_56": S("run:secondary_all.csv", "ceiling_vs_rtm", F3, {"indicator_id": "OVERALL", "method": "rtm"}),
    "cmp_spearman_methods": K("cmp_spearman_methods_overall_skill", F2),
    "cmp_spearman_ind": K("cmp_spearman_indicators_primary", F2),
    "cmp_best45": {**K("cmp_best_method_45", "method")},
    "cmp_best56": {**K("cmp_best_method_56", "method")},
    "t56_compare": T("run:comparison_45_vs_56.csv", "method", MAIN_METHODS,
                     [("Skill 4→5 (CV)", "skill_45", F3, {"indicator_id": "OVERALL"}),
                      ("Skill 5→6 (test)", "skill_56", F3, {"indicator_id": "OVERALL"}),
                      ("Change", "skill_change", F3, {"indicator_id": "OVERALL"})], label="Method"),
    "cov_split": K("coverage_split_primary", F3), "cov_weighted": K("coverage_weighted_primary", F3),
    "cov_normalised": K("coverage_normalised_primary", F3),
    "t56_coverage": T("run:coverage.csv", ["variant", "method"], None,
                      [("n", "n", "int"), ("Coverage", "coverage", F3), ("Mean width (pp)", "mean_width", F1)],
                      where={"subset": "all", "indicator_id": "OVERALL"}, label="Variant · method",
                      sort=(["variant", "method"], True)),
    "breaks_any": K("breaks_any", "int"), "breaks_shift": K("breaks_shift", "int"),
    "breaks_slope": K("breaks_slope", "int"), "breaks_scale": K("breaks_scale", "int"),
    "t56_breaks": T("run:structural_break.csv", "indicator_id", None,
                    [("Δ mean change (pp)", "shift_diff", F1), ("Δ RTM slope", "slope_diff", F2),
                     ("Δ RMS error (pp)", "scale_diff", F1)], where={"break_any": "True"}, label="Indicator",
                    sort=("indicator_id", True)),
    "n_surprises": K("surprises_primary", "int"), "n_surprise_tests": K("surprise_tests_primary", "int"),
    "surprise_overlap": K("surprise_overlap_naive", "int"), "naive_k": K("naive_k", "int"),
    "naive_se": K("naive_mean_se", F2), "bh_se": K("bh_mean_se", F2),
    "t56_surprises": T("run:surprises_{primary}.csv", ["district_uid", "indicator_id"], None,
                       [("Forecast", "y_pred", F1), ("Observed", "y_true", F1), ("z", "z", F2), ("BH q", "q_normal", F3),
                        ("Unit type", "unit_type", "text")], where={"surprise": "True"}, label="District · indicator",
                       sort=("p_normal", True), head=15),
    "r2_ind": K("decomp_r2_indicator", F3), "r2_dist": K("decomp_r2_district", F3), "r2_add": K("decomp_r2_additive", F3),
    "t56_hard": T("run:forecastability_indicators.csv", "indicator_id", None,
                  [("Noise ceiling", "ceiling", F3), ("Normalised skill", "skill", F3)], label="Indicator",
                  sort=("skill", True), head=6),
    "fig56_skill": FIG("run:skill_by_indicator_all.png", "Skill vs persistence by indicator, NFHS-5 → NFHS-6 (frozen models)"),
    "fig56_skill_stable": FIG("run:skill_by_indicator_stable.png", "The same on the stable-boundary subset"),
    "fig56_compare": FIG("run:comparison_45_vs_56.png", "Per-indicator skill in NFHS-4 → 5 CV vs NFHS-5 → 6"),
    "fig56_coverage": FIG("run:coverage.png", "Conformal coverage on NFHS-6"),
    "fig56_break": FIG("run:structural_break.png", "Structural-break check (exploratory)"),
    "fig56_surprises": FIG("run:surprises.png", "BH-FDR surprises vs naive top-k"),
    "fig56_map_skill": FIG("run:map_skill.png", "Forecast skill by district"),
    "fig56_map_ceiling": FIG("run:map_ceiling.png", "Noise ceiling by district"),
}


# ---------------------------------------------------------------------
# Context and sources
# ---------------------------------------------------------------------
class Ctx:
    def __init__(self, cfg: dict, mode: str, run_dir: Path | None, report_path: Path):
        self.cfg, self.mode = cfg, mode
        self.outputs = Path(cfg["abs_paths"]["outputs_dir"])
        self.run_dir = Path(run_dir) if run_dir else None
        self.prefix = "DRYRUN_" if mode == "dryrun" else ""
        self.report_dir = report_path.parent
        from .post_run import choose_primary
        self.primary = choose_primary(pd.read_csv(self.outputs / "results_nfhs4to5.csv"),
                                      cfg["gate_c"]["primary_method"])
        self._cache: dict[Path, object] = {}

    def resolve(self, src: str) -> Path | None:
        src = src.replace("{primary}", self.primary)
        kind, name = src.split(":", 1)
        if kind == "out":
            return self.outputs / name
        if kind == "run":
            return None if self.mode == "prerun" else self.run_dir / f"{self.prefix}{name}"
        raise ValueError(f"unknown source {src}")

    def load(self, path: Path):
        if path not in self._cache:
            if path.suffix == ".json":
                self._cache[path] = json.loads(path.read_text(encoding="utf-8"))
            else:
                self._cache[path] = pd.read_csv(path, dtype=str, keep_default_na=False)
        return self._cache[path]


def fmt_value(v, fmt: str) -> str:
    if fmt == "text":
        return str(v)
    if fmt == "method":
        return METHOD_LABELS.get(str(v), str(v))
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(x):
        return "n/a"
    if fmt == "int":
        return f"{int(round(x)):d}"
    return fmt.format(x)


def _filter(df: pd.DataFrame, where: dict, primary: str) -> pd.DataFrame:
    for k, v in where.items():
        if k.startswith("@"):
            continue
        v = str(v).replace("{primary}", primary)
        col = df[k].astype(str)
        try:
            fv = float(v)
            num = pd.to_numeric(col, errors="coerce")
            df = df[(col == v) | (num.notna() & np.isclose(num.fillna(np.inf), fv))]
        except ValueError:
            df = df[col == v]
    return df


def _scalar(spec: dict, ctx: Ctx) -> str:
    path = ctx.resolve(spec["src"])
    data = ctx.load(path)
    if isinstance(data, dict):
        v = data[spec["col"]]
        return fmt_value(len(v) if spec["agg"] == "len" else v, spec["fmt"])
    df = _filter(data, spec["where"], ctx.primary)
    if spec["agg"] == "count":
        return fmt_value(len(df), spec["fmt"])
    if len(df) != 1:
        raise ValueError(f"{path.name}: {len(df)} rows match {spec['where']} (need exactly 1)")
    return fmt_value(df[spec["col"]].iloc[0], spec["fmt"])


def _cell(df: pd.DataFrame, col: str, fmt: str) -> str:
    if df.empty:
        return "n/a"
    if col.startswith("ci:"):
        base = col[3:]
        return f"[{fmt_value(df[base + '_lo'].iloc[0], fmt)}, {fmt_value(df[base + '_hi'].iloc[0], fmt)}]"
    return fmt_value(df[col].iloc[0], fmt)


def _table(spec: dict, ctx: Ctx) -> str:
    base = _filter(ctx.load(ctx.resolve(spec["src"])), spec["where"], ctx.primary)
    keys = spec["row_col"] if isinstance(spec["row_col"], list) else [spec["row_col"]]
    if spec["rows"] is None:
        rows = base.copy()
        if spec["sort"]:
            col, asc = spec["sort"]
            cols = col if isinstance(col, list) else [col]
            for c in cols:
                num = pd.to_numeric(rows[c], errors="coerce")
                if num.notna().all():
                    rows[c + "__num"] = num
            rows = rows.sort_values([c + "__num" if c + "__num" in rows else c for c in cols], ascending=asc)
        if spec["head"]:
            rows = rows.head(spec["head"])
        row_keys = [tuple(r) for r in rows[keys].itertuples(index=False)]
    else:
        row_keys = [(r,) for r in spec["rows"]]
    head = [spec["label"]] + [c[0] for c in spec["cols"]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    if not row_keys:
        return "*(none)*"
    for rk in row_keys:
        label = " · ".join(METHOD_LABELS.get(str(x), str(x)) for x in rk)
        cells = []
        for c in spec["cols"]:
            h, col, fmt = c[0], c[1], c[2]
            extra = dict(c[3]) if len(c) > 3 else {}
            src = extra.pop("@src", None)
            df = base if src is None else _filter(ctx.load(ctx.resolve(src)), spec["where"], ctx.primary)
            df = _filter(df, {**dict(zip(keys, rk)), **extra}, ctx.primary)
            cells.append(_cell(df, col, fmt))
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def _verdict(spec: dict, ctx: Ctx) -> str:
    df = _filter(ctx.load(ctx.resolve(spec["src"])), spec["where"], ctx.primary)
    lo, hi = float(df[spec["lo"]].iloc[0]), float(df[spec["hi"]].iloc[0])
    return spec["neg"] if hi < 0 else spec["pos"] if lo > 0 else spec["zero"]


def _fig(spec: dict, ctx: Ctx) -> str:
    path = ctx.resolve(spec["src"])
    if not path.exists():
        raise FileNotFoundError(path)
    rel = Path(os.path.relpath(path, ctx.report_dir)).as_posix()
    return f"![{spec['caption']}]({rel})"


STATUS = {
    "prerun": ("**Status: before Gate B.** NFHS-4 → NFHS-5 results are final. Every NFHS-6 number is marked "
               f"`{PENDING}` and is filled in automatically by `scripts/post_run.py` after the one-shot evaluation."),
    "final": ("**Status: final.** All NFHS-6 numbers below were filled in automatically by `scripts/post_run.py` "
              "from `outputs/post_run/` after the one-shot evaluation, and checked by `scripts/check_report_numbers.py`."),
    "dryrun": ("**DRYRUN — NOT NFHS-6 RESULTS.** The 'NFHS-6' numbers in this file come from the NFHS-4 → NFHS-5 "
               "leave-one-state-out predictions used as a stand-in, only to test the report pipeline."),
}


def render_slot(name: str, ctx: Ctx) -> str:
    spec = SLOTS[name]
    kind = spec["kind"]
    if kind == "status":
        return STATUS[ctx.mode]
    if kind == "primary":
        return METHOD_LABELS.get(ctx.primary, ctx.primary)
    if ctx.mode == "prerun" and spec.get("src", "").startswith("run:"):
        return PENDING
    return {"scalar": _scalar, "table": _table, "verdict": _verdict, "fig": _fig}[kind](spec, ctx)


def render(template: str, ctx: Ctx) -> tuple[str, dict]:
    values = {}

    def sub(m):
        name = m.group(1)
        if name not in SLOTS:
            raise KeyError(f"unknown slot {{{{{name}}}}}")
        text = render_slot(name, ctx)
        spec = SLOTS[name]
        src = ctx.resolve(spec["src"]) if "src" in spec else None
        values[name] = {"text": text, "source": None if src is None else Path(os.path.relpath(src, ctx.report_dir)).as_posix()}
        return f"<!--slot:{name}-->{text}<!--/slot-->"
    return SLOT_RE.sub(sub, template), values


def report_paths(cfg: dict, mode: str, out_dir: Path | None = None) -> tuple[Path, Path, Path | None]:
    root = Path(cfg["abs_paths"]["outputs_dir"]).parent
    if mode == "dryrun":
        d = Path(out_dir) if out_dir else root / cfg["post_run"]["dryrun_dir"]
        return d / "DRYRUN_report.md", d / "DRYRUN_report_values.json", d
    run_dir = (Path(out_dir) if out_dir else root / cfg["post_run"]["out_dir"]) if mode == "final" else None
    return root / "docs" / "report.md", root / "docs" / "report_values.json", run_dir


def build(cfg: dict, mode: str, out_dir: Path | None = None, template: Path | None = None) -> tuple[Path, Path]:
    """Render the report for `mode` ('prerun', 'final', 'dryrun'). Returns (report path, values path)."""
    if mode not in STATUS:
        raise ValueError(mode)
    root = Path(cfg["abs_paths"]["outputs_dir"]).parent
    template = template or root / "docs" / "report_template.md"
    report, values_path, run_dir = report_paths(cfg, mode, out_dir)
    if mode != "dryrun" and run_dir is not None and "dryrun" in str(run_dir).lower():
        raise ValueError(f"refusing to build the real report from a DRYRUN folder: {run_dir}")
    report.parent.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(cfg, mode, run_dir, report)
    text, values = render(template.read_text(encoding="utf-8"), ctx)
    header = (f"<!-- GENERATED by scripts/build_report.py (mode={mode}) from docs/report_template.md. "
              "Edit the template, not this file. -->\n")
    write_text(report, header + text)
    write_text(values_path, json.dumps({"mode": mode, "primary_method": ctx.primary, "slots": values}, indent=1,
                                       ensure_ascii=False) + "\n")
    return report, values_path


def check(report: Path, values: Path, cfg: dict, mode: str) -> list[str]:
    from .report_check import check_report
    return check_report(report, cfg, mode)
