"""Gate C analyses: conformal coverage, FDR surprises, per-district forecastability (DECISIONS D34).

AFTER the one-shot NFHS-6 run (Gate B), on its predictions:

    .venv\\Scripts\\python scripts\\gate_c_analysis.py --predictions outputs\\predictions_nfhs5to6.parquet

BEFORE it (development; never touches NFHS-6): calibrate on the NFHS-4 -> NFHS-5
LOSO residuals of a random half of the states and evaluate on the other half:

    .venv\\Scripts\\python scripts\\gate_c_analysis.py --dev

Calibration is always outputs/predictions_nfhs4to5.parquet (LOSO residuals,
harmonised training set). Outputs go to outputs/gate_c/ (or outputs/gate_c_dev/):
coverage.csv, intervals.parquet, surprises_<method>.csv, surprise_summary.json,
forecastability_districts.csv, forecastability_indicators.csv,
forecastability_decomposition.json, map_*.png, gate_c_summary.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfhs_forecast import conformal, evaluate, forecastability, surprise  # noqa: E402
from nfhs_forecast.config import load_config, region_of, write_csv, write_json, write_text  # noqa: E402
from nfhs_forecast.report import md_table  # noqa: E402

PRIMARY_CANDIDATES = ["national_trend", "state_trend", "rtm", "fay_herriot", "fh_rtm", "elastic_net", "lightgbm"]


def choose_primary(results_csv: Path, rule: str) -> str:
    """Pre-registered rule (D34): lowest pooled LOSO MSE on NFHS-4 -> NFHS-5 among the primary methods."""
    if rule != "auto":
        return rule
    r = pd.read_csv(results_csv)
    r = r[(r.indicator_id == "OVERALL") & r.method.isin(PRIMARY_CANDIDATES)]
    return str(r.sort_values(["rmse", "method"]).iloc[0].method)


def dev_split(cal: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = np.array(sorted(cal.state.unique()))
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(states, size=len(states) // 2, replace=False))
    return cal[cal.state.isin(pick)].copy(), cal[~cal.state.isin(pick)].copy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--predictions", default=None, help="target predictions (format of predictions_nfhs5to6)")
    ap.add_argument("--calibration", default=None, help="default: <outputs>/predictions_nfhs4to5.parquet")
    ap.add_argument("--dev", action="store_true", help="development mode on NFHS-4 -> NFHS-5 LOSO residuals only")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    g = cfg["gate_c"]
    outd = cfg["abs_paths"]["outputs_dir"]
    cal_path = Path(args.calibration) if args.calibration else outd / "predictions_nfhs4to5.parquet"
    cal = pd.read_parquet(cal_path)
    if args.dev == bool(args.predictions):
        print("Give exactly one of --predictions <file> or --dev.")
        return 2
    if args.dev:
        cal, target = dev_split(cal, g["dev_seed"])
        label = "DEVELOPMENT: calibrated on a random half of states, NFHS-4 -> NFHS-5 LOSO; evaluated on the other half"
        out = Path(args.out_dir) if args.out_dir else outd / "gate_c_dev"
    else:
        target = pd.read_parquet(args.predictions)
        label = f"target: {Path(args.predictions).name}; calibration: {cal_path.name}"
        out = Path(args.out_dir) if args.out_dir else outd / "gate_c"
    out.mkdir(parents=True, exist_ok=True)
    primary = choose_primary(outd / "results_nfhs4to5.csv", g["primary_method"])
    alpha = g["alpha"]
    methods = sorted(set(g["conformal_methods"]) | {primary})

    # C1 conformal intervals
    iv = conformal.conformal_intervals(cal, target, alpha, methods=methods, region=region_of(cfg),
                                       weight_clip=g["weight_clip"], seed=cfg["random_seed"])
    iv.to_parquet(out / "intervals.parquet", index=False)
    cov = conformal.coverage_table(iv, alpha)
    if "stable_unit" in iv.columns:
        cov_s = conformal.coverage_table(iv[iv.stable_unit.astype(bool)], alpha).assign(subset="stable")
        cov = pd.concat([cov.assign(subset="all"), cov_s], ignore_index=True)
    write_csv(cov, out / "coverage.csv", float_format="%.4f")

    # C2 surprises
    summaries = []
    for m in sorted({primary, "rtm"}):
        tab, summ = surprise.detect_surprises(cal, target, m, q=g["fdr_q"], floor_share=g["sigma_floor_share"],
                                              naive_k_min=g["naive_k_min"])
        write_csv(tab, out / f"surprises_{m}.csv", float_format="%.4f")
        summaries.append(summ)
    write_json({"primary_method": primary, "label": label, "summaries": summaries}, out / "surprise_summary.json")

    # C3 forecastability
    dist = forecastability.district_table(target, primary)
    ind = forecastability.indicator_table(target, primary)
    dec = forecastability.decompose(target, primary)
    write_csv(dist, out / "forecastability_districts.csv", float_format="%.4f")
    write_csv(ind, out / "forecastability_indicators.csv", float_format="%.4f")
    write_json(dec, out / "forecastability_decomposition.json")
    panel_path = cfg["abs_paths"]["processed_dir"] / "panel_model.parquet"
    n_mapped = 0
    if panel_path.exists():
        pm = pd.read_parquet(panel_path, columns=["district_uid", "district_code"])
        codes = pm.dropna(subset=["district_code"]).groupby("district_uid").district_code.first()
        s = dist.set_index("district_uid")
        n_mapped = forecastability.plot_map(s.skill, codes, cfg["abs_paths"]["geojson"], out / "map_skill.png",
                                            f"Normalised forecast skill by district ({primary})",
                                            "1 - mean normalised squared error")
        forecastability.plot_map(s.ceiling, codes, cfg["abs_paths"]["geojson"], out / "map_ceiling.png",
                                 "Noise ceiling by district", "1 - mean noise share")

    ov = cov[(cov.indicator_id == "OVERALL") & (cov.get("subset", "all") == "all")] if "subset" in cov else \
        cov[cov.indicator_id == "OVERALL"]
    L = ["# Gate C analyses", "", label, "", f"Primary forecaster (rule D34): `{primary}`; alpha = {alpha}; "
         f"BH q = {g['fdr_q']}.", "", "## C1 Conformal coverage (OVERALL)", "",
         md_table(ov[["variant", "method", "n", "coverage", "nominal", "mean_width"]], floatfmt="{:.3f}"), "",
         "## C2 Surprises", "", md_table(pd.DataFrame(summaries), floatfmt="{:.3f}"), "",
         "## C3 Forecastability", "",
         f"Decomposition of log normalised squared error ({dec['n']} pairs, {dec['districts']} districts, "
         f"{dec['indicators']} indicators): indicator-only R² {dec['r2_indicator_only']:.3f}, "
         f"district-only R² {dec['r2_district_only']:.3f}, additive R² {dec['r2_additive']:.3f} "
         f"(unique: indicator {dec['unique_indicator']:.3f}, district {dec['unique_district']:.3f}).", "",
         f"Districts mapped: {n_mapped} Census-2011 polygons.", "",
         "Hardest indicators:", "", md_table(ind.tail(5), floatfmt="{:.3f}"), ""]
    write_text(out / "gate_c_summary.md", "\n".join(L) + "\n")
    print(f"Gate C outputs in {out} (primary method {primary}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
