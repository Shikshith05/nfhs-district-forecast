"""FINAL OUT-OF-TIME TEST: predict NFHS-6 from NFHS-5 with the FROZEN models and score it.

DO NOT RUN until the team has reviewed the pipeline and git-tagged the freeze, e.g.

    git tag -a freeze-v1 -m "Models frozen before NFHS-6 evaluation"
    .venv\\Scripts\\python scripts\\evaluate_nfhs6.py --confirm-frozen

Requires outputs/models/frozen_models.pkl and data/processed/panel_model.parquet
from scripts/run_pipeline.py. Nothing is refitted: every method uses
parameters estimated on NFHS-4 -> NFHS-5 and only NFHS-5 values as inputs.
  persistence   y6 = y5
  state_trend   y6 = y5 + state mean change NFHS-4 -> NFHS-5 (extrapolated)
  rtm           y6 = a + b * y5, (a, b) fitted on NFHS-4 -> NFHS-5
  eb_shrinkage  y6 = EB-shrunk y5 toward the NFHS-5 state mean
  elastic_net / lightgbm   frozen models applied to NFHS-5 features
The oracle state trend is not computed (it would need NFHS-6 values).
Run it once. If anything is changed after seeing NFHS-6 results, say so in
DECISIONS.md and report both runs.

"Run once" is enforced (DECISIONS D26): the script refuses to run unless the
git tag `evaluation.freeze_tag` (freeze-v1) exists and points at HEAD and the
working tree is clean, and it refuses if outputs/nfhs6_evaluation_run.json
already exists. That marker is created before any NFHS-6 value is scored and
records the commit id.

Exit codes: 0 ok, 2 not confirmed, 3 training-data hash mismatch, 4 no NFHS-6
rows, 5 already run, 6 freeze/git check failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from nfhs_forecast import evaluate, freeze_guard, models  # noqa: E402
from nfhs_forecast.config import config_hash, load_config, write_csv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-frozen", action="store_true",
                    help="required: confirms the freeze was reviewed and tagged")
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--panel", default=None, help="default: <processed_dir>/panel_model.parquet")
    ap.add_argument("--models", default=None, help="default: <models_dir>/frozen_models.pkl")
    ap.add_argument("--out-dir", default=None, help="default: <outputs_dir>")
    ap.add_argument("--allow-hash-mismatch", action="store_true",
                    help="score even if the NFHS-4/5 training data differ from the frozen data")
    ap.add_argument("--repo", default=str(REPO), help="git working tree to check (default: this repo)")
    args = ap.parse_args(argv)
    if not args.confirm_frozen:
        print("Refusing to run: this is the one-shot NFHS-6 test. Review the pipeline, git-tag the "
              "freeze, then re-run with --confirm-frozen.")
        return 2

    cfg = load_config(args.config)
    P = cfg["abs_paths"]
    panel_path = Path(args.panel) if args.panel else P["processed_dir"] / "panel_model.parquet"
    models_path = Path(args.models) if args.models else P["models_dir"] / "frozen_models.pkl"
    out = Path(args.out_dir) if args.out_dir else P["outputs_dir"]
    out.mkdir(parents=True, exist_ok=True)

    try:
        freeze_guard.check_not_already_run(out)
    except freeze_guard.AlreadyRunError as e:
        print(f"ABORT: {e}")
        return 5
    tag = cfg["evaluation"].get("freeze_tag", "freeze-v1")
    try:
        git_info = freeze_guard.check_freeze(args.repo, tag)
    except freeze_guard.FreezeCheckError as e:
        print(f"ABORT: {e}")
        return 6

    frozen = models.load_frozen(models_path)
    panel = pd.read_parquet(panel_path)

    # The training data must be exactly what the models were frozen on.
    train_panel = panel[panel["round"].isin([4, 5])
                        & panel.district_uid.isin(frozen.core_districts)
                        & panel.indicator_id.isin(frozen.indicators)]
    h = models.training_data_hash(train_panel)
    if h != frozen.train_hash:
        msg = f"training-data hash {h} != frozen {frozen.train_hash}: NFHS-4/5 inputs changed since the freeze"
        if not args.allow_hash_mismatch:
            print("ABORT: " + msg)
            return 3
        print("WARNING: " + msg)
    config_match = config_hash(cfg) == frozen.config_hash
    if not config_match:
        print("WARNING: config.yaml changed since the freeze (frozen models are used regardless).")

    # Claim the one run BEFORE any NFHS-6 value is scored.
    marker = freeze_guard.claim_run(out, {
        **git_info, "frozen_created_utc": frozen.created_utc, "train_hash": frozen.train_hash,
        "hash_check": h == frozen.train_hash, "config_hash_match": config_match,
        "allow_hash_mismatch": bool(args.allow_hash_mismatch)})

    # Build NFHS-5 -> NFHS-6. Round-6 values are read ONLY to score predictions.
    # Test units (harmonised set) and the stable-boundary subset were fixed at the
    # freeze from NFHS-5 data and NFHS-6 metadata only (DECISIONS D28).
    test_units = frozen.test_units or frozen.core_districts
    stable_units = set(frozen.stable_units or frozen.core_districts)
    # Spatial neighbours use the geometry frozen with the models (Census-2011 polygons).
    data = models.build_transition(panel, 5, 6, frozen.indicators, test_units, cfg,
                                   for_training=False, adjacency=frozen.fitted.spec.adjacency)
    if data.next is None:
        freeze_guard.finish_run(marker, outcome="aborted: no NFHS-6 rows in the panel")
        print("ABORT: no NFHS-6 rows in the panel.")
        return 4
    # Predictions get a copy without the round-6 matrices, so they cannot see NFHS-6.
    preds = models.predict_methods(frozen.fitted, data.inputs_only(), cfg, include_eb=True)
    preds = evaluate.attach_truth(preds, data)
    preds = evaluate.annotate_units(preds, frozen.units, stable_units)
    preds.to_parquet(out / "predictions_nfhs5to6.parquet", index=False)

    # Pre-registered tables (DECISIONS D32), on the harmonised set and the stable subset.
    ev = cfg["evaluation"]
    kw = dict(bootstrap_reps=ev["bootstrap_reps"], ci_level=ev["ci_level"], seed=cfg["random_seed"])
    outputs = ["predictions_nfhs5to6.parquet"]
    for suffix, sub in (("", preds), ("_stable", preds[preds.stable_unit])):
        res = evaluate.metric_table(sub, evaluate.HONEST_METHODS, **kw)
        tests = evaluate.paired_tests(sub, **kw)
        tables = {f"results_nfhs5to6{suffix}.csv": res,
                  f"secondary_nfhs5to6{suffix}.csv": evaluate.secondary_metrics(sub, **kw),
                  f"paired_tests_nfhs5to6{suffix}.csv": tests,
                  f"paired_summary_nfhs5to6{suffix}.csv": evaluate.paired_summary(tests)}
        for name, table in tables.items():
            write_csv(table, out / name, float_format="%.4f")
            outputs.append(name)
        if suffix == "":
            results = res
    evaluate.plot_skill(results, out / "skill_by_indicator_nfhs6.png",
                        "Skill vs persistence, NFHS-5 → NFHS-6 (frozen models)")
    outputs.append("skill_by_indicator_nfhs6.png")
    freeze_guard.finish_run(marker, outcome="scored", outputs=outputs, n_test_units=len(test_units),
                            n_stable_units=len(stable_units))

    overall = results[results.indicator_id == "OVERALL"].set_index("method")
    print("OVERALL (pooled over indicators), NFHS-5 -> NFHS-6, frozen models")
    print(overall[["n", "mae", "rmse", "skill", "skill_lo", "skill_hi"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
