"""Run the whole base pipeline with one command.

    .venv\\Scripts\\python scripts\\run_pipeline.py

Steps: load -> clean -> match indicators -> match districts + NFHS-5 audit ->
panel -> codebook -> data report -> SEs + boundary harmonisation ->
leave-one-state-out CV on NFHS-4 -> NFHS-5 (harmonised set and stable
subset; all pre-registered methods; secondary metrics; paired tests) ->
seen-state CV (reference) -> results + plot -> freeze models.

NFHS-6 values are used only for loading, cleaning, the data report and the
audit. This script never scores NFHS-6; that is scripts/evaluate_nfhs6.py,
which the team runs after reviewing and git-tagging the freeze.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from nfhs_forecast import (clean, evaluate, harmonise, load, match, models, noise, report,  # noqa: E402
                           spatial)
from nfhs_forecast.config import (config_hash, ensure_dirs, load_config, write_csv,  # noqa: E402
                                   write_json)


def step(i: int, n: int, msg: str) -> None:
    print(f"[{i}/{n}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--bootstrap", type=int, default=None, help="override bootstrap reps")
    ap.add_argument("--skip-cv", action="store_true", help="stop after the data report")
    ap.add_argument("--hash-only", "--data-only", dest="hash_only", action="store_true",
                    help="rebuild data/processed and the harmonisation outputs, print the training hash, "
                         "and stop: no CV, no re-freeze (used by scripts/preflight.py)")
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
    write_csv(variants, proc / "indicator_variants.csv", index=False)

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
    write_csv(name_map, proc / "district_name_map.csv", index=False)
    write_csv(name_map[name_map.core], proc / "matched_districts.csv", index=False)
    write_csv(name_map[~name_map.core], proc / "unmatched_districts.csv", index=False)
    write_csv(audit_pairs, proc / "audit_nfhs5_pairs.csv", index=False)
    core_uids = sorted(name_map.loc[name_map.core, "district_uid"])
    print(f"      core districts: {len(core_uids)}; NFHS-5 audit within 0.5: "
          f"{audit_summary['within_0_5_agreement_rate']:.1%}")

    step(5, N, "Building tidy panel and indicator codebook")
    panel = match.build_panel_long(l45, d6, variants, name_map)
    panel.to_parquet(proc / "panel_long.parquet", index=False)
    codebook = match.build_codebook(cross, panel, set(core_uids), cfg, audit_by_ind)
    write_csv(codebook, proc / "indicator_codebook.csv", index=False)
    included = codebook.loc[codebook.include, "indicator_id"].tolist()
    print(f"      indicators: {int(codebook.comparable_all_rounds.sum())} comparable, "
          f"{len(included)} included")

    step(6, N, "Writing outputs/data_report.md")
    report.write_data_report(outd / "data_report.md", log45=log45, log6=log6, panel=panel,
                             name_map=name_map, codebook=codebook, variants=variants,
                             audit_summary=audit_summary, audit_pairs=audit_pairs,
                             audit_by_district=audit_by, cfg=cfg)
    write_json({"nfhs45": log45, "nfhs6": log6, "audit": audit_summary}, proc / "cleaning_log.json")
    if args.skip_cv:
        print(f"Done (skip-cv) in {time.time() - t0:.0f}s")
        return 0

    step(7, N, "Assumed standard errors (ASSUMPTIONS) and boundary harmonisation")
    hcfg = cfg.get("harmonise", {"enabled": False})
    if hcfg.get("enabled"):
        node_panel = panel[panel.district_uid.notna() & panel.indicator_id.isin(included)]
        node_panel = noise.add_se(node_panel, codebook, cfg)
        codes = (panel[panel["round"] == 5].dropna(subset=["district_code"])
                 .groupby("district_uid").district_code.first().to_dict())
        supplied = harmonise.load_supplied(cfg)
        crosswalk = harmonise.build_crosswalk(name_map, cfg, codes)
        crosswalk, problems = harmonise.apply_supplied_lineage(crosswalk, supplied["lineage"], name_map, cfg)
        model_panel, units, crosswalk = harmonise.harmonise_panel(node_panel, name_map, crosswalk, cfg, supplied)
        review = harmonise.review_table(crosswalk, units, problems, cfg)
        hsum = harmonise.summarise(units, name_map, cfg)
        write_csv(crosswalk, outd / "crosswalk.csv")
        write_csv(review, outd / "crosswalk_review.csv")
        write_csv(units, outd / "harmonised_units.csv")
        harmonise.write_report(outd / "harmonisation_report.md", hsum, crosswalk, review, units, supplied, cfg)
        print(f"      units: {hsum['train_units']} train (+{hsum['train_units_added']}), "
              f"{hsum['test_units']} test (+{hsum['test_units_added']}), {hsum['stable_units']} stable; "
              f"{len(review)} crosswalk cases for review")
    else:
        model_panel = noise.add_se(panel[panel.district_uid.isin(core_uids) & panel.indicator_id.isin(included)],
                                   codebook, cfg)
        units = pd.DataFrame({"unit_uid": core_uids, "unit_type": "identity", "parent_uid": core_uids,
                              "train_unit": True, "test_unit": True, "stable_unit": True})
    model_panel.to_parquet(proc / "panel_model.parquet", index=False)
    train_uids = sorted(units.loc[units.train_unit, "unit_uid"])
    stable_uids = sorted(units.loc[units.stable_unit & units.train_unit, "unit_uid"])
    # ---- NFHS-6 is removed here; everything below trains on NFHS-4 -> NFHS-5 only ----
    train_panel = model_panel[model_panel["round"].isin([4, 5]) & model_panel.district_uid.isin(train_uids)].copy()
    models.assert_no_round6(train_panel, "run_pipeline")
    if args.hash_only:
        print(f"TRAIN_HASH={models.training_data_hash(train_panel)}")
        print(f"TRAIN_UNITS={len(train_uids)}")
        return 0

    step(8, N, f"Cross-validation on NFHS-4 -> NFHS-5 ({len(included)} indicators)")
    ev = cfg["evaluation"]
    reps = ev["bootstrap_reps"] if args.bootstrap is None else args.bootstrap
    kw = dict(bootstrap_reps=reps, ci_level=ev["ci_level"], seed=cfg["random_seed"])
    domain = codebook.set_index("indicator_id").domain
    adjacency = spatial.load_adjacency(P["geojson"])
    sets_all, sets_sec = cfg["models"]["feature_sets"], cfg["models"]["feature_sets_secondary"]

    def write_tables(preds, suffix):
        res = evaluate.metric_table(preds, evaluate.METHOD_ORDER, **kw)
        res.insert(1, "domain", res.indicator_id.map(domain).fillna("all"))
        tests = evaluate.paired_tests(preds, **kw)
        write_csv(res, outd / f"results_nfhs4to5{suffix}.csv", float_format="%.4f")
        write_csv(evaluate.secondary_metrics(preds, **kw), outd / f"secondary_nfhs4to5{suffix}.csv",
                  float_format="%.4f")
        write_csv(tests, outd / f"paired_tests_nfhs4to5{suffix}.csv", float_format="%.4f")
        write_csv(evaluate.paired_summary(tests), outd / f"paired_summary_nfhs4to5{suffix}.csv",
                  float_format="%.4f")
        return res

    cv_sets = [("", train_uids, sets_all, "harmonised training set" if hcfg.get("enabled") else "core districts")]
    if hcfg.get("enabled"):
        cv_sets.append(("_stable", stable_uids, sets_sec, "stable-boundary subset"))
    for suffix, uids, sets, label in cv_sets:
        print(f"      LOSO CV on the {label}: {len(uids)} districts, feature sets {sets}")
        preds = evaluate.loso_cv(train_panel[train_panel.district_uid.isin(uids)], included, uids, cfg,
                                 verbose=False, adjacency=adjacency, feature_sets=sets)
        preds.to_parquet(outd / f"predictions_nfhs4to5{suffix}.parquet", index=False)
        res = write_tables(preds, suffix)
        if suffix == "":
            results = res
            write_csv(evaluate.deff_sensitivity(preds, cfg), outd / "sensitivity_deff.csv", float_format="%.4f")
            evaluate.plot_skill(res, outd / "skill_by_indicator.png",
                                f"Skill vs persistence, NFHS-4 → NFHS-5, leave-one-state-out CV "
                                f"({len(uids)} districts)")
    print(f"      seen-state CV (optimistic reference): {ev['seen_state_folds']} folds")
    parents = units.set_index("unit_uid").parent_uid
    preds = evaluate.seen_state_cv(train_panel, included, train_uids, cfg, groups=parents,
                                   adjacency=adjacency, n_splits=ev["seen_state_folds"])
    preds.to_parquet(outd / "predictions_nfhs4to5_seenstate.parquet", index=False)
    write_tables(preds, "_seenstate")

    step(9, N, "Freezing models trained on all NFHS-4 -> NFHS-5 data")
    trans = models.build_transition(train_panel, 4, 5, included, train_uids, cfg, adjacency=adjacency)
    spec = evaluate.feature_spec(trans, included, sets_all, adjacency)
    frozen = models.freeze(train_panel, trans, cfg, spec, config_hash(cfg), units=units)
    models.save_frozen(frozen, P["models_dir"] / "frozen_models.pkl")
    manifest = {"created_utc": frozen.created_utc, "train_hash": frozen.train_hash,
                "config_hash": frozen.config_hash, "indicators": frozen.indicators,
                "n_core_districts": len(frozen.core_districts), "n_test_units": len(frozen.test_units),
                "n_stable_units": len(frozen.stable_units), "harmonised": bool(hcfg.get("enabled")),
                "feature_sets": sets_all, "methods": evaluate.HONEST_METHODS,
                "versions": frozen.versions, "trained_on": "NFHS-4 -> NFHS-5 only", "nfhs6_scored": False}
    write_json(manifest, P["models_dir"] / "manifest.json")

    overall = results[results.indicator_id == "OVERALL"].set_index("method")
    print("\nOVERALL (pooled over indicators), NFHS-4 -> NFHS-5 LOSO CV")
    print(overall[["mae", "rmse", "skill", "skill_lo", "skill_hi"]].round(3).to_string())
    mean = results[results.indicator_id == "MEAN"].set_index("method")
    print("\nMEAN skill over indicators (unweighted)")
    print(mean[["skill", "skill_lo", "skill_hi"]].round(3).to_string())
    print(f"\nDone in {time.time() - t0:.0f}s. NFHS-6 has NOT been scored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
