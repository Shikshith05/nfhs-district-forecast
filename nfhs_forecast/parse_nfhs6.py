"""STUB: independent parser for the IIPS NFHS-6 district fact-sheet PDFs. NOT IMPLEMENTED.

The pipeline currently uses the community compilation by Patra (2026),
data/raw/nfhs6/all_india_nfhs6_district.csv (see docs/DATA_SOURCES.md).
That file has known issues handled in clean.py / match.py: '*' cells,
truncated indicator text, state-sheet rows for Chandigarh and Lakshadweep
with different numbering, a merged 'Dadra & Nagar Haveli And Daman & Diu'
block, and a row-level small_sample_flag that does not say which round's
value is small-sample. The NFHS-5 audit also found Maharajganj and Mahoba
(Uttar Pradesh) swapped relative to India.csv.

TODO (to be implemented later)
------------------------------
1. Download the official district PDFs from https://iipsindia.ac.in/content/nfhs-projects
   into data/raw/nfhs6/pdf/ (keep the originals; record file hashes).
2. Extract tables (e.g. pdfplumber), keeping per-cell markers: '*' for
   suppressed and '( )' for 25-49 unweighted cases, separately for the
   NFHS-6 and NFHS-5 columns.
3. Emit the same columns as the community CSV plus small_sample_nfhs6 and
   small_sample_nfhs5, so clean.clean_nfhs6 can consume either source.
4. Audit against the community CSV: report cell-level agreement and resolve
   the Maharajganj/Mahoba swap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_district_pdf(path: str | Path) -> pd.DataFrame:
    """TODO: parse one IIPS NFHS-6 district fact sheet PDF (see module docstring)."""
    raise NotImplementedError("Independent NFHS-6 PDF parsing is not implemented yet.")
