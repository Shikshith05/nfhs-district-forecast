# Data sources

All raw files live in `data/raw/` and are tracked in git (each is under 50 MB),
so the pipeline runs straight after cloning. Nothing in `data/raw/` is edited
by the code; all corrections happen in `nfhs_forecast/clean.py` and are listed
in `DECISIONS.md`.

| File | Rounds | Size | Source |
|---|---|---|---|
| `data/raw/nfhs4_5/India.csv` | NFHS-4 (2015-16), NFHS-5 (2019-21) | 10.8 MB, 73,632 rows | SaiSiddhardhaKalla/NFHS |
| `data/raw/nfhs6/all_india_nfhs6_district.csv` | NFHS-6 (2023-24), provisional | 9.8 MB, 66,519 rows | Patra (2026) |
| `data/raw/nfhs6/<state>_nfhs6_district.csv` (35 files) | NFHS-6 per state | 0.03-1.1 MB each | Patra (2026); used only for spot checks |
| `data/raw/nfhs6/SOURCE/` (`CITATION.cff`, `LICENSE`, `README.md`) | n/a | small | Attribution files of the Patra (2026) repository; keep these |
| `data/raw/geo/Districts.geojson` | Census 2011 district polygons | 28.0 MB, 641 features | Distributed with SaiSiddhardhaKalla/NFHS |
| `docs/CodeFlow_Jackfruit_OnePager.docx` | n/a | 10 KB | Team CodeFlow project one-pager |

## NFHS-4 / NFHS-5: `India.csv`

- **Source:** https://github.com/SaiSiddhardhaKalla/NFHS
- **What it is:** values parsed from the IIPS NFHS-5 *district* fact sheets,
  which print each NFHS-5 (2019-21) estimate next to the comparable NFHS-4
  (2015-16) estimate. Columns: `State, ST_CEN_CD, District, DISTRICT,
  DT_CEN_CD, Category, Indicator, NFHS 5, NFHS 4`.
- **Codes:** `ST_CEN_CD` / `DT_CEN_CD` / `DISTRICT` follow Census 2011. About 65
  districts created after 2011 have no code (70 rows of district names in this
  copy); split districts carry their parent's 2011 key.
- **Known quirks** (handled in code): `na` and suppressed `*` cells were written
  as `0`; trailing spaces in names; hyphens dropped from text (`2024` for
  `20-24`); one mislabelled block (Saraikela-Kharsawan under Sahibganj); a
  placeholder district `Data Not Available`; one conflicting duplicate row.
- **Original publisher:** International Institute for Population Sciences
  (IIPS) and ICF. *National Family Health Survey (NFHS-5), 2019-21: District
  Fact Sheets.* Mumbai: IIPS. http://rchiips.org/nfhs/

## NFHS-6: `all_india_nfhs6_district.csv`

- **Source:** Patra, N. K. (2026). *NFHS-6 Provisional District-Level Data
  (India): Independent Compilation.* GitHub.
  https://github.com/nabinpatra007/nfhs6-district-data-india (version 1.0.0,
  MIT licence for code; see `data/raw/nfhs6/SOURCE/`).
- **What it is:** parsed from the IIPS **provisional** NFHS-6 State and District
  Fact Sheets (2023-24). 35 States/UTs (Manipur was not surveyed), 712 district
  entries, about 93 indicators each. Columns: `state, district, category,
  indicator_no, indicator, nfhs6_value, nfhs5_value, small_sample_flag,
  boundary_change_single_round`.
- **Known quirks** (handled in code): `*` suppressed cells; Chandigarh and
  Lakshadweep rows come from the *state* sheet (different `indicator_no`);
  `Dadra & Nagar Haveli And Daman & Diu` merges several districts (380 rows);
  truncated and footnote-variant indicator text; `small_sample_flag` is per row,
  not per round.
- **Provisional:** IIPS may revise these values. Re-run the pipeline if the
  source CSV is updated, and record the version used.
- **Cite both:**
  - International Institute for Population Sciences (IIPS). *National Family
    Health Survey-6 (NFHS-6), 2023-24: State and District Fact Sheets.*
    Mumbai: IIPS. https://iipsindia.ac.in/content/nfhs-projects
  - Patra, N. K. (2026). *NFHS-6 Provisional District-Level Data (India):
    Independent Compilation.* GitHub.

## District boundaries: `Districts.geojson`

- Census 2011 district polygons (641 features; CRS84) with properties
  `DISTRICT, ST_NM, ST_CEN_CD, DT_CEN_CD, censuscode, Dist_name`. Obtained with
  the SaiSiddhardhaKalla/NFHS data. The pipeline reads only the properties, to
  attach the Census 2011 `censuscode` as `district_code`. Not yet used for maps
  or spatial features (out of scope).

## What was removed from the pasted folder

`nfhs6-district-data-india-main/scripts/` (the original PDF-parsing scripts:
`extract_nfhs6.py`, `consolidate_data.py`, `run_all_states.py`) was deleted as
instructed; they remain available in the upstream repository above.
