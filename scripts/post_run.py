"""One command for everything after the one-shot NFHS-6 evaluation (DECISIONS D37-D40).

    .venv\\Scripts\\python scripts\\post_run.py            # real run: needs the Gate B outputs
    .venv\\Scripts\\python scripts\\post_run.py --dryrun   # DRYRUN on NFHS-4 -> NFHS-5 LOSO predictions

Real mode reads outputs/predictions_nfhs5to6.parquet and requires
outputs/nfhs6_evaluation_run.json with status "completed". It writes
outputs/post_run/ (tables, figures, key numbers), renders docs/report.md
with every NFHS-6 number filled in, checks every number in the report
against the result files, and adds round 6 to outputs/release/.

--dryrun uses the NFHS-4 -> NFHS-5 leave-one-state-out predictions as a
stand-in in the same format, writes ONLY to outputs/dryrun/ with every file
named and marked DRYRUN, and renders outputs/dryrun/DRYRUN_report.md. It
never touches docs/report.md or the release.

Nothing is refitted; the frozen models are not loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from nfhs_forecast import post_run as pr  # noqa: E402
from nfhs_forecast.config import load_config  # noqa: E402


def unit_codes(cfg: dict) -> pd.Series | None:
    """Census-2011 code per unit: the processed panel if present, else the release unit table."""
    panel = cfg["abs_paths"]["processed_dir"] / "panel_model.parquet"
    if panel.exists():
        pm = pd.read_parquet(panel, columns=["district_uid", "district_code"])
        return pm.dropna(subset=["district_code"]).groupby("district_uid").district_code.first()
    rel = Path(cfg["abs_paths"]["outputs_dir"]) / "release" / "units.csv"
    if rel.exists():
        u = pd.read_csv(rel, dtype={"district_code": str})
        return u.dropna(subset=["district_code"]).set_index("unit_uid").district_code
    return None


def check_real_inputs(target_path: Path, marker: Path, target: pd.DataFrame) -> str | None:
    if "dryrun" in str(target_path).lower():
        return f"{target_path} looks like a DRYRUN file; real mode refuses it"
    if "run_mode" in target.columns and (target.run_mode == pr.DRYRUN).any():
        return f"{target_path} contains DRYRUN rows; real mode refuses it"
    if not marker.exists():
        return f"{marker} not found: run Gate B (scripts/gate_b.ps1) first"
    status = json.loads(marker.read_text(encoding="utf-8")).get("status")
    if status != "completed":
        return f"run log status is '{status}', not 'completed'"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--dryrun", action="store_true", help="stand-in run on NFHS-4 -> NFHS-5 predictions")
    ap.add_argument("--predictions", default=None, help="real mode: default outputs/predictions_nfhs5to6.parquet")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-report", action="store_true", help="tables and figures only")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = Path(args.config).resolve().parent
    outd = Path(cfg["abs_paths"]["outputs_dir"])
    pcfg = cfg["post_run"]

    cal = pd.read_parquet(outd / "predictions_nfhs4to5.parquet")
    res45 = pd.read_csv(outd / "results_nfhs4to5.csv")
    units = pd.read_csv(outd / "harmonised_units.csv")
    if args.dryrun:
        target = pr.dryrun_standin(cal, units)
        out = pr.Out(Path(args.out_dir) if args.out_dir else root / pcfg["dryrun_dir"], dryrun=True)
        label56 = "DRYRUN stand-in (NFHS-4 → NFHS-5 CV)"
        target.to_parquet(out.path("predictions_nfhs5to6.parquet"), index=False)
        out.md("README.md", "# DRYRUN outputs\n\nEvery file in this folder was produced by "
               "`scripts/post_run.py --dryrun` from the NFHS-4 → NFHS-5 leave-one-state-out predictions, "
               "used as a stand-in in the NFHS-5 → NFHS-6 predictions format. They test the post-run "
               "pipeline and report and are NOT NFHS-6 results. Every file name starts with DRYRUN_, every "
               "table has run_mode = DRYRUN, and every figure is watermarked. `scripts/post_run.py` in real "
               "mode refuses these files, and the real report is checked never to refer to them.\n")
        print("DRYRUN: stand-in = NFHS-4 -> NFHS-5 LOSO predictions; outputs are NOT NFHS-6 results.")
    else:
        tpath = Path(args.predictions) if args.predictions else outd / "predictions_nfhs5to6.parquet"
        if not tpath.exists():
            print(f"{tpath} not found: run Gate B first (or use --dryrun).")
            return 2
        target = pd.read_parquet(tpath)
        problem = check_real_inputs(tpath, outd / "nfhs6_evaluation_run.json", target)
        if problem:
            print("REFUSED: " + problem)
            return 3
        out = pr.Out(Path(args.out_dir) if args.out_dir else root / pcfg["out_dir"], dryrun=False)
        label56 = "NFHS-5 → NFHS-6 (frozen models)"

    summary = pr.run(target, cal, res45, units, cfg, out, codes=unit_codes(cfg), label56=label56)
    print(f"Post-run tables and figures in {out.root} ({len(set(out.written))} files); primary method "
          f"{summary['primary_method']}.")
    if args.skip_report:
        return 0
    from nfhs_forecast import report_build as rb
    mode = "dryrun" if args.dryrun else "final"
    report_path, values_path = rb.build(cfg, mode, out_dir=out.root)
    problems = rb.check(report_path, values_path, cfg, mode)
    if problems:
        print(f"REPORT CHECK FAILED ({len(problems)}):\n  " + "\n  ".join(problems[:30]))
        return 4
    print(f"Report: {report_path} (all numbers checked against the result files).")
    if not args.dryrun:
        from nfhs_forecast import release
        rel = release.make_release(cfg, include_nfhs6=True)
        print(f"Release with NFHS-6 values: {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
