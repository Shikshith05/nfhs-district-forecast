"""Spot-check the community NFHS-6 CSV against the official IIPS district PDFs (Task A4).

    .venv\\Scripts\\python scripts\\spot_check_nfhs6.py            # random sample (config pdf_check)
    .venv\\Scripts\\python scripts\\spot_check_nfhs6.py --all      # every row of every PDF supplied

Put the IIPS NFHS-6 district fact-sheet PDFs in data/raw/pdfs/ with a
manifest.csv (file,state,district; names as in the NFHS-6 CSV, see
docs/NFHS6_PDF_CHECK.md). The script samples district x indicator rows at
random (fixed seed) from the districts whose PDFs are present, parses the
PDFs (text layer only, no OCR, no guessing), and reports how often the CSV
and the PDF agree, for the NFHS-6 and the NFHS-5 column.

Outputs: outputs/nfhs6_spot_check.csv (one row per sampled pair) and
outputs/nfhs6_spot_check.md (agreement rates with 95% Wilson CIs).
This is a data audit: the values are not used for modelling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfhs_forecast import clean, load  # noqa: E402
from nfhs_forecast import parse_nfhs6 as pp  # noqa: E402
from nfhs_forecast.clean import normalise_name  # noqa: E402
from nfhs_forecast.config import load_config, write_csv, write_text  # noqa: E402
from nfhs_forecast.report import md_table  # noqa: E402


def summarise(res: pd.DataFrame, infos: pd.DataFrame, n_requested: int, seed: int, mode: str) -> str:
    comp = res[res.status == "compared"]
    L = ["# NFHS-6 spot-check against the IIPS PDFs", "",
         f"Mode: {mode}; seed {seed}; pairs requested {n_requested}; pairs drawn {len(res)}; "
         f"compared {len(comp)}.", "",
         "A pair agrees if the community CSV and the PDF give the same value (within the tolerance), "
         "or both show `*` (suppressed), or both are not available. Pairs that could not be matched "
         "or whose PDF row was ambiguous are counted separately and never guessed.", "",
         "| column | agree | compared | rate | 95% CI |", "|---|---|---|---|---|"]
    for col in ("nfhs6", "nfhs5"):
        k, n = int(comp[f"agree_{col}"].sum()) if len(comp) else 0, len(comp)
        lo, hi = pp.wilson_ci(k, n)
        L.append(f"| {col.upper()} | {k} | {n} | {k / n:.1%} | [{lo:.1%}, {hi:.1%}] |" if n else
                 f"| {col.upper()} | 0 | 0 | n/a | n/a |")
    if len(comp):
        both = comp.agree_nfhs6 & comp.agree_nfhs5
        lo, hi = pp.wilson_ci(int(both.sum()), len(comp))
        L.append(f"| both | {int(both.sum())} | {len(comp)} | {both.mean():.1%} | [{lo:.1%}, {hi:.1%}] |")
    L += ["", "## Pair status", "", md_table(res.status.value_counts().rename("pairs").reset_index()), "",
          "## Files", "", md_table(infos.fillna("")), ""]
    bad = comp[~(comp.agree_nfhs6 & comp.agree_nfhs5)] if len(comp) else comp
    if len(bad):
        L += ["## Disagreements", "",
              md_table(bad[["state", "district", "indicator_no", "indicator", "csv_nfhs6", "pdf_nfhs6",
                            "csv_nfhs5", "pdf_nfhs5"]].head(50)), ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--pdf-dir", default=None, help="default: config pdf_check.pdf_dir")
    ap.add_argument("--n", type=int, default=None, help="pairs to sample (default: config pdf_check.sample_size)")
    ap.add_argument("--all", action="store_true", help="compare every row of every supplied PDF")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pcfg = cfg["pdf_check"]
    root = Path(args.config).resolve().parent
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else root / pcfg["pdf_dir"]
    out = Path(args.out_dir) if args.out_dir else cfg["abs_paths"]["outputs_dir"]
    seed = int(pcfg.get("seed", cfg["random_seed"]))
    n = args.n or int(pcfg["sample_size"])

    if not (pdf_dir / "manifest.csv").exists():
        print(f"No {pdf_dir / 'manifest.csv'}: put the IIPS district PDFs in {pdf_dir} and list them in "
              "manifest.csv (file,state,district). See docs/NFHS6_PDF_CHECK.md. Nothing was checked.")
        return 1
    parsed, infos = pp.parse_pdf_folder(pdf_dir, tuple(pcfg["value_columns"]))
    if parsed.empty:
        print("No PDF could be parsed:\n" + infos.to_string())
        return 1

    d6, _ = clean.clean_nfhs6(load.read_nfhs6(cfg["abs_paths"]["nfhs6_csv"]), cfg)
    d6 = d6[["state", "district", "indicator_no", "indicator", "nfhs6_value", "nfhs5_value"]]
    have = {(normalise_name(s), normalise_name(d)) for s, d in
            infos.loc[infos.status.eq("parsed"), ["state", "district"]].itertuples(index=False)}
    keyed = list(zip(d6.state.map(normalise_name), d6.district.map(normalise_name)))
    pool = d6[[k in have for k in keyed]].reset_index(drop=True)
    unknown = have - set(keyed)
    if unknown:
        print(f"WARNING: manifest districts not found in the NFHS-6 CSV: {sorted(unknown)}")
    if pool.empty:
        print("None of the parsed PDFs matches a district in the NFHS-6 CSV; check manifest names.")
        return 1
    rng = np.random.default_rng(seed)
    sample = pool if args.all else pool.iloc[np.sort(rng.choice(len(pool), size=min(n, len(pool)),
                                                                replace=False))]
    res = pp.compare_pairs(sample, parsed, tol=float(pcfg["tolerance"]))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(res, out / "nfhs6_spot_check.csv")
    write_text(out / "nfhs6_spot_check.md", summarise(res, infos, len(pool) if args.all else n, seed,
                                                      "all rows" if args.all else "random sample"))
    comp = res[res.status == "compared"]
    print(f"Compared {len(comp)} of {len(res)} pairs. Agreement NFHS-6: "
          f"{comp.agree_nfhs6.mean():.1%}, NFHS-5: {comp.agree_nfhs5.mean():.1%}" if len(comp) else
          f"No pair could be compared ({len(res)} drawn); see outputs/nfhs6_spot_check.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
