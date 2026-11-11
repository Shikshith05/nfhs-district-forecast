"""Write the harmonised panel and codebook release to outputs/release/ (DECISIONS D40).

    .venv\\Scripts\\python scripts\\make_release.py      # NFHS-4 and NFHS-5 only (before Gate B)

scripts/post_run.py adds NFHS-6 after the one-shot evaluation. Needs
data/processed/ from scripts/run_pipeline.py (--skip-cv is enough for the data).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nfhs_forecast import release  # noqa: E402
from nfhs_forecast.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out = release.make_release(load_config(args.config), include_nfhs6=False,
                               out_dir=Path(args.out_dir) if args.out_dir else None)
    print(f"Release (NFHS-4/5 only) written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
