# Sources and licence

## Underlying data

- **NFHS-4 and NFHS-5:** International Institute for Population Sciences (IIPS) and ICF, National Family
  Health Survey district fact sheets (http://rchiips.org/nfhs/), via the community parse
  https://github.com/SaiSiddhardhaKalla/NFHS (`India.csv`). That repository shows **no licence** (checked
  2026-10-23); its README states the data are sourced from IIPS.
- **NFHS-6 (provisional):** IIPS, NFHS-6 (2023-24) State and District Fact Sheets
  (https://iipsindia.ac.in/content/nfhs-projects), via Patra, N. K. (2026), *NFHS-6 Provisional
  District-Level Data (India): Independent Compilation*, v1.0.0
  (https://github.com/nabinpatra007/nfhs6-district-data-india). Its MIT licence covers code only; its data
  notice states that the licence does not apply to NFHS-6 / IIPS data.
- **Boundaries:** Census 2011 district polygons distributed with the SaiSiddhardhaKalla/NFHS data.

## This release

The tables are derived by Team CodeFlow (cleaning, matching, harmonisation, assumed standard errors).
The NFHS estimates remain the work of IIPS and the Ministry of Health and Family Welfare, Government of
India, and are subject to their terms of use. The project repository has no licence file of its own: the
team must choose one for its code and derived tables before publishing them, and must cite IIPS and both
community sources when this release is reused.

**NFHS-6 values are provisional** and were parsed from PDFs by a third party; check them against the
official fact sheets before relying on any single value.
