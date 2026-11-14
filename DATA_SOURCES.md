# Data Sources

All data used in this project are public. NFHS-6 values are provisional.

## Survey rounds

| Round | Period | Used for |
|---|---|---|
| NFHS-4 | 2015-16 | Training (NFHS-4 to NFHS-5 transition) |
| NFHS-5 | 2019-21 | Training target and test input |
| NFHS-6 (provisional) | 2023-24 | One-shot test only; never used for training or tuning |

## Links

| Data | Link |
|---|---|
| Official source: IIPS NFHS district fact sheets | https://rchiips.org/nfhs |
| NFHS-6 district compilation (Patra, N. K., 2026; community-parsed from IIPS fact sheets, MIT licence on code) | https://github.com/nabinpatra007/nfhs6-district-data-india |
| This project (code, data, results) | https://github.com/Shikshith05/nfhs-district-forecast |

## Files in this repository

| Path | Contents |
|---|---|
| `data/raw/nfhs4_5/` | NFHS-4 and NFHS-5 district values (`India.csv`, community parse of IIPS fact sheets) |
| `data/raw/nfhs6/` | NFHS-6 provisional district values (Patra compilation) |
| `data/raw/geo/` | Census-2011 district boundaries (`Districts.geojson`) |
| `data/lineage/` | Parent-child district lineage crosswalk and Census-2011 populations used for harmonisation |
| `outputs/release/` | Harmonised district panel and indicator codebook |

## Notes

- The NFHS-6 compilation is unofficial and provisional. IIPS and MoHFW do not endorse it.
- Standard errors are assumed (design effect 2.0, sensitivity 1.5 to 3), not published.
- Manipur has no NFHS-6 district data.
- The harmonised panel combines split or merged districts using Census-2011 population weights. NFHS-6 outcomes were never used to choose weights.
- Please credit IIPS and the NFHS programme, and the Patra compilation, when reusing the data.
