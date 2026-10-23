# Spot-checking NFHS-6 values against the IIPS PDFs

The NFHS-6 district values come from a community compilation (Patra 2026).
`scripts/spot_check_nfhs6.py` compares a random sample of its district ×
indicator values with the official IIPS district fact sheets. It uses
`nfhs_forecast/parse_nfhs6.py`, an independent parser that reads the PDF text
layer and never guesses a value.

## What you need to do

1. Download the NFHS-6 district fact sheets (PDF) from IIPS
   (https://www.iipsindia.ac.in/content/nfhs-projects, or the NFHS site
   linked there). Any subset of districts works. For a 200-pair check, about
   30–40 districts spread over several states is plenty. Choose the districts
   **before** looking at any values, e.g. every 20th district in
   `data/raw/nfhs6/all_india_nfhs6_district.csv`, so the check stays random.
2. Put the PDFs in `data/raw/pdfs/`. Keep the original files unchanged.
3. Create `data/raw/pdfs/manifest.csv` with one row per PDF:

   ```csv
   file,state,district
   wayanad.pdf,Kerala,Wayanad
   ```

   `state` and `district` must be spelled as in the NFHS-6 CSV (columns
   `state`, `district`). The script reports any name it cannot find.
4. Run:

   ```powershell
   .venv\Scripts\python scripts\spot_check_nfhs6.py          # 200 random pairs (config pdf_check)
   .venv\Scripts\python scripts\spot_check_nfhs6.py --all    # every row of every PDF
   ```

5. Read `outputs/nfhs6_spot_check.md`. It shows the agreement rate for the
   NFHS-6 and NFHS-5 columns with 95% Wilson CIs, the status of every pair
   and the disagreements. Look up each disagreement in the PDF by hand.
   Commit both outputs. If the rate is low, log it in DECISIONS.md before
   Gate B.

## How the parser behaves

- **Text layer only.** A scanned PDF without a text layer is reported as
  `no_text_layer`. Nothing is OCR-ed.
- **Rows.** A row starts at `<number>. <text>` and may wrap. It is accepted
  only when its last two tokens are values (`12.3`, `(12.3)` = 25–49 cases,
  `*` = suppressed, `na`) and the token before them is not a number.
  Otherwise the row is `ambiguous` or `incomplete` and is excluded from the
  comparison.
- **Matching.** Indicators are matched to the CSV by normalised text (exact,
  or a unique prefix of at least 25 characters for truncated text), never by
  number alone.
- **Agreement.** A pair agrees if the values are within 0.05
  (`pdf_check.tolerance`), or both are `*`, or both are not available.
- **Column order.** The order of the value columns is `pdf_check.value_columns`
  (`[nfhs6, nfhs5]`). If the fact sheets have more columns (e.g. urban/rural),
  set this list to match.

The sample seed (`pdf_check.seed` = 2026) and size (200) were fixed before
any PDF was seen (DECISIONS D33).
