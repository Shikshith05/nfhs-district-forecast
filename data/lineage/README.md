# Supplied inputs for boundary harmonisation

`nfhs_forecast/harmonise.py` uses only lineage the repository can support
(name matches, India.csv NFHS-4 availability, NFHS-6 boundary flags). To
recover more districts, put any of the three files below in this folder.
Missing files are fine. Each one is read when `scripts/run_pipeline.py` runs.
Lines starting with `#` are ignored. The `*.template.csv` files show the
columns and are never read. Copy one to the real file name and fill it in.

Do **not** derive any weight or lineage from NFHS-6 outcome values (HARD RULE 1).

## 1. `lineage_supplied.csv`: official parent → child district lineage

| column | meaning |
|---|---|
| `transition` | `5to6` (NFHS-5 district → NFHS-6 district) or `4to5` (Census-2011 district → NFHS-5 district) |
| `state` | state name as in `outputs/crosswalk.csv` |
| `parent_district` | `5to6`: NFHS-5 name (`parent_name` in the crosswalk). `4to5`: Census-2011 name (GeoJSON `Dist_name`) |
| `child_district` | `5to6`: NFHS-6 name (`child_name`). `4to5`: NFHS-5 name |
| `relation` | `unchanged`, `renamed`, `split`, `merged`, `transfer` (informational; the graph decides the units) |
| `weight` | `4to5` only, when a child has several parents: the share of the child's population from this parent (shares sum to 1 per child). Otherwise leave it empty |
| `source` | e.g. `LGD modification report 2019-2024, downloaded 2026-10-25` or a gazette notification number |
| `confidence` | `high` (official source), `medium`, `low` |

For a split, list **every** child including the parent's remnant. For example,
Vellore → Vellore, Vellore → Ranipet and Vellore → Tirupathur are three
`5to6` rows. Supplied rows replace the repo-derived rows for the same child.
A name the pipeline cannot find is reported in `outputs/crosswalk_review.csv`.

Source: Local Government Directory, https://lgdirectory.gov.in → Download
Directory → "Download Only Modifications" (entity: districts; one state at a
time; date range 2019-01-01 to 2024-12-31 for `5to6`, 2015-01-01 to
2021-12-31 for `4to5`). The site needs a captcha, so it cannot be fetched
automatically.

## 2. `population.csv`: populations for weights

| column | meaning |
|---|---|
| `vintage` | `nfhs5` or `nfhs6` (which boundary set the district belongs to) |
| `state`, `district` | names as in `outputs/crosswalk.csv` (`parent_name` for nfhs5, `child_name` for nfhs6) |
| `population` | population count (total population unless `basis` says otherwise) |
| `basis` | `total`, `women_15_49`, `children_under5`, ... (informational) |
| `reference_year` | e.g. 2011 |
| `source` | e.g. `Census 2011 PCA, sub-districts re-aggregated per LGD` |
| `estimated` | `true` if the population was re-aggregated or projected. The weights then carry `harmonise.weight_cv` uncertainty |

Needed for every NFHS-5 and NFHS-6 district in a changed component. Without
populations, a split is scored by inheritance: each NFHS-6 child is compared
with its parent's forecast, flagged, with extra variance.

## 3. `nfhs4_census2011_values.csv`: NFHS-4 values on Census-2011 boundaries

| column | meaning |
|---|---|
| `state`, `district` | Census-2011 name (the `parent_district` used in `4to5` lineage rows) |
| `census_code` | Census-2011 district code (optional, for checking) |
| `indicator_id` | an id from `data/processed/indicator_codebook.csv` (the 23 included ones) |
| `value` | NFHS-4 district fact-sheet value (%) |
| `source` | e.g. `IIPS NFHS-4 district fact sheet, <file name>` |

With `4to5` lineage rows, these let the 132 NFHS-5 districts whose
boundaries changed between NFHS-4 and NFHS-5 enter training. Their NFHS-4
value is inherited from the parent, flagged, with extra variance (DECISIONS
D29).
