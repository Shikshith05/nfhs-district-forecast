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
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from nfhs_forecast import evaluate, models  # noqa: E402
from nfhs_forecast.config import config_hash, load_config  # noqa: E402


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
    if config_hash(cfg) != frozen.config_hash:
        print("WARNING: config.yaml changed since the freeze (frozen models are used regardless).")

    # Build NFHS-5 -> NFHS-6. Round-6 values are read ONLY to score predictions.
    data = models.build_transition(panel, 5, 6, frozen.indicators, frozen.core_districts, cfg,
                                   for_training=False)
    if data.next is None:
        print("ABORT: no NFHS-6 rows in the panel.")
        return 4
    # Predictions get a copy without the round-6 matrices, so they cannot see NFHS-6.
    inputs_only = models.Transition(5, 6, data.prev, data.se_prev, data.meta)
    preds = models.predict_methods(frozen.fitted, inputs_only, cfg, include_eb=True)
    preds = evaluate.attach_truth(preds, data)
    preds.to_parquet(out / "predictions_nfhs5to6.parquet", index=False)

    results = evaluate.metric_table(preds, evaluate.HONEST_METHODS,
                                    bootstrap_reps=cfg["evaluation"]["bootstrap_reps"],
                                    ci_level=cfg["evaluation"]["ci_level"], seed=cfg["random_seed"])
    results.to_csv(out / "results_nfhs5to6.csv", index=False, float_format="%.4f")
    evaluate.plot_skill(results, out / "skill_by_indicator_nfhs6.png",
                        "Skill vs persistence, NFHS-5 → NFHS-6 (frozen models)")
    with open(out / "nfhs6_evaluation_run.json", "w", encoding="utf-8") as f:
        json.dump({"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "frozen_created_utc": frozen.created_utc, "train_hash": frozen.train_hash,
                   "hash_check": h == frozen.train_hash}, f, indent=1)

    overall = results[results.indicator_id == "OVERALL"].set_index("method")
    print("OVERALL (pooled over indicators), NFHS-5 -> NFHS-6, frozen models")
    print(overall[["n", "mae", "rmse", "skill", "skill_lo", "skill_hi"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
