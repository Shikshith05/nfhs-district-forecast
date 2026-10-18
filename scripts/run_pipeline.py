"""Run the whole base pipeline with one command.

    .venv\\Scripts\\python scripts\\run_pipeline.py

Steps: load -> clean -> match indicators -> match districts + NFHS-5 audit ->
panel -> codebook -> data report -> SEs -> leave-one-state-out CV on
NFHS-4 -> NFHS-5 -> results + plot -> freeze models.

NFHS-6 values are used only for loading, cleaning, the data report and the
audit. This script never scores NFHS-6; that is scripts/evaluate_nfhs6.py,
which the team runs after reviewing and git-tagging the freeze.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from nfhs_forecast import clean, evaluate, load, match, models, noise, report  # noqa: E402
from nfhs_forecast.config import config_hash, ensure_dirs, load_config  # noqa: E402


def step(i: int, n: int, msg: str) -> None:
    print(f"[{i}/{n}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--bootstrap", type=int, default=None, help="override bootstrap reps")
    ap.add_argument("--skip-cv", action="store_true", help="stop after the data report")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    P = cfg["abs_paths"]
    proc, outd = P["processed_dir"], P["outputs_dir"]
    N = 9

    step(1, N, "Loading raw data")
    raw45 = load.read_nfhs45(P["nfhs45_csv"])
    raw6 = load.read_nfhs6(P["nfhs6_csv"])
    geo = load.read_geo_properties(P["geojson"])

    step(2, N, "Cleaning")
    l45, log45 = clean.clean_nfhs45(raw45, cfg, geo)
    d6, log6 = clean.clean_nfhs6(raw6, cfg)

    step(3, N, "Matching indicators by normalised text")
    variants, cross = match.build_indicator_matching(l45, d6, cfg)
    if cross.attrs.get("core_not_found"):
        raise SystemExit(f"Core indicators not found in NFHS-6 text: {cross.attrs['core_not_found']}")
    variants.to_csv(proc / "indicator_variants.csv", index=False)

    step(4, N, "Matching districts and auditing NFHS-5 values across sources")
    name_map = match.match_districts(l45, d6, cfg)
    audit_pairs, audit_summary, audit_by, audit_by_ind = match.audit_nfhs5(
        l45, d6, variants, name_map, cross, cfg["indicators"]["min_audit_agreement"],
        district_indicators=set(cross.loc[cross.core, "indicator_id"]))
    name_map = match.finalise_district_status(name_map, audit_by, cfg)
    nm_cols = ["state", "district", "district_uid", "state_key", "state_nfhs6", "district_nfhs6", "state_nfhs45",
               "district_nfhs45", "method", "score", "boundary_change_nfhs6", "has_nfhs4", "has_nfhs5",
               "audit_pairs", "audit_exact", "audit_within_0_5", "core", "exclusion_reason"]
    name_map = name_map[nm_cols].sort_values(["state", "district"])
    name_map.to_csv(proc / "district_name_map.csv", index=False)
    name_map[name_map.core].to_csv(proc / "matched_districts.csv", index=False)
    name_map[~name_map.core].to_csv(proc / "unmatched_districts.csv", index=False)
    audit_pairs.to_csv(proc / "audit_nfhs5_pairs.csv", index=False)
    core_uids = sorted(name_map.loc[name_map.core, "district_uid"])
    print(f"      core districts: {len(core_uids)}; NFHS-5 audit within 0.5: "
          f"{audit_summary['within_0_5_agreement_rate']:.1%}")

    step(5, N, "Building tidy panel and indicator codebook")
    panel = match.build_panel_long(l45, d6, variants, name_map)
    panel.to_parquet(proc / "panel_long.parquet", index=False)
    codebook = match.build_codebook(cross, panel, set(core_uids), cfg, audit_by_ind)
    codebook.to_csv(proc / "indicator_codebook.csv", index=False)
    included = codebook.loc[codebook.include, "indicator_id"].tolist()
    print(f"      indicators: {int(codebook.comparable_all_rounds.sum())} comparable, "
          f"{len(included)} included")

    step(6, N, "Writing outputs/data_report.md")
    report.write_data_report(outd / "data_report.md", log45=log45, log6=log6, panel=panel,
                             name_map=name_map, codebook=codebook, variants=variants,
                             audit_summary=audit_summary, audit_pairs=audit_pairs,
                             audit_by_district=audit_by, cfg=cfg)
    with open(proc / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump({"nfhs45": log45, "nfhs6": log6, "audit": audit_summary}, f, indent=1, default=str)
    if args.skip_cv:
        print(f"Done (skip-cv) in {time.time() - t0:.0f}s")
        return 0

    step(7, N, "Adding assumed standard errors (config.yaml noise: ASSUMPTIONS)")
    model_panel = panel[panel.district_uid.isin(core_uids) & panel.indicator_id.isin(included)]
    model_panel = noise.add_se(model_panel, codebook, cfg)
    model_panel.to_parquet(proc / "panel_model.parquet", index=False)
    # ---- NFHS-6 is removed here; everything below trains on NFHS-4 -> NFHS-5 only ----
    train_panel = model_panel[model_panel["round"].isin([4, 5])].copy()
    models.assert_no_round6(train_panel, "run_pipeline")

    step(8, N, f"Leave-one-state-out CV on NFHS-4 -> NFHS-5 ({len(included)} indicators)")
    preds = evaluate.loso_cv(train_panel, included, core_uids, cfg)
    preds.to_parquet(outd / "predictions_nfhs4to5.parquet", index=False)
    reps = cfg["evaluation"]["bootstrap_reps"] if args.bootstrap is None else args.bootstrap
    results = evaluate.metric_table(preds, evaluate.METHOD_ORDER, bootstrap_reps=reps,
                                    ci_level=cfg["evaluation"]["ci_level"], seed=cfg["random_seed"])
    domain = codebook.set_index("indicator_id").domain
    results.insert(1, "domain", results.indicator_id.map(domain).fillna("all"))
    results.to_csv(outd / "results_nfhs4to5.csv", index=False, float_format="%.4f")
    evaluate.deff_sensitivity(preds, cfg).to_csv(outd / "sensitivity_deff.csv", index=False,
                                                 float_format="%.4f")
    evaluate.plot_skill(results, outd / "skill_by_indicator.png",
                        f"Skill vs persistence, NFHS-4 → NFHS-5, leave-one-state-out CV "
                        f"({len(core_uids)} districts)")

    step(9, N, "Freezing models trained on all NFHS-4 -> NFHS-5 data")
    trans = models.build_transition(train_panel, 4, 5, included, core_uids, cfg)
    spec = evaluate.feature_spec(trans, included)
    frozen = models.freeze(train_panel, trans, cfg, spec, config_hash(cfg))
    models.save_frozen(frozen, P["models_dir"] / "frozen_models.pkl")
    manifest = {"created_utc": frozen.created_utc, "train_hash": frozen.train_hash,
                "config_hash": frozen.config_hash, "indicators": frozen.indicators,
                "n_core_districts": len(frozen.core_districts), "versions": frozen.versions,
                "trained_on": "NFHS-4 -> NFHS-5 only", "nfhs6_scored": False}
    with open(P["models_dir"] / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    overall = results[results.indicator_id == "OVERALL"].set_index("method")
    print("\nOVERALL (pooled over indicators), NFHS-4 -> NFHS-5 LOSO CV")
    print(overall[["mae", "rmse", "skill", "skill_lo", "skill_hi"]].round(3).to_string())
    print(f"\nDone in {time.time() - t0:.0f}s. NFHS-6 has NOT been scored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
