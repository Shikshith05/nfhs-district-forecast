"""Check every number in a rendered report against the result files (DECISIONS D38).

    .venv\\Scripts\\python scripts\\check_report_numbers.py                     # docs/report.md
    .venv\\Scripts\\python scripts\\check_report_numbers.py --report outputs\\dryrun\\DRYRUN_report.md

The mode (prerun / final / dryrun) is read from the report's header. See
nfhs_forecast/report_check.py for the list of checks. Exit code 1 on any problem.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nfhs_forecast.config import load_config  # noqa: E402
from nfhs_forecast.report_check import check_report  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--report", default=str(REPO / "docs" / "report.md"))
    args = ap.parse_args(argv)
    report = Path(args.report)
    m = re.search(r"mode=(prerun|final|dryrun)", report.read_text(encoding="utf-8")[:300])
    if not m:
        print(f"{report}: no generated-report header; render it with scripts/build_report.py first.")
        return 1
    problems = check_report(report, load_config(args.config), m.group(1))
    for p in problems:
        print("  FAIL " + p)
    print(f"{report} (mode {m.group(1)}): {len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
