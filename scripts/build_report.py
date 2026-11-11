"""Render the report from docs/report_template.md and check it (DECISIONS D38).

    .venv\\Scripts\\python scripts\\build_report.py                # before Gate B: NFHS-6 slots PENDING
    .venv\\Scripts\\python scripts\\build_report.py --mode final   # after post_run.py (it calls this itself)

Writes docs/report.md and docs/report_values.json (the source of every
number), then runs the same checks as scripts/check_report_numbers.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nfhs_forecast import report_build as rb  # noqa: E402
from nfhs_forecast.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--mode", choices=["prerun", "final"], default="prerun")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    report, values = rb.build(cfg, args.mode)
    problems = rb.check(report, values, cfg, args.mode)
    for p in problems:
        print("  FAIL " + p)
    print(f"{report}: {'CHECK FAILED' if problems else 'all numbers checked'} ({len(problems)} problem(s)).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
