# Summary

## Gate B preparation (2026-10-23)

Everything that does not need NFHS-6 results is done (D35–D40, all added after the
pre-registrations D32/D34 and labelled so). **NFHS-6 has still not been scored.**

- `scripts/preflight.py`: PASS/WARN/FAIL checks; currently PASSED with 2 WARNs (Bilaspur
  lineage; D34 committed together with the development outputs).
- `scripts/gate_b.ps1`: pre-flight → YES → tag → one-shot evaluation → commit; stops on errors.
- `scripts/post_run.py`: one command after Gate B for every table, figure, the filled
  report and the release. `--dryrun` tested it end to end on the NFHS-4 → NFHS-5
  predictions (`outputs/dryrun/`, every file marked DRYRUN, not NFHS-6 results).
- `docs/report.md`: generated from `docs/report_template.md`; NFHS-4 → NFHS-5 numbers
  filled, NFHS-6 slots `[[NFHS-6 PENDING]]`; every number checked by
  `scripts/check_report_numbers.py`.
- `outputs/release/`: harmonised panel (NFHS-4/5 only until Gate B), units, codebook,
  crosswalk, data dictionary, sources and licence.
- `docs/HUMAN_CHECKLIST.md` (what to check before Gate B) and `docs/DEFENCE_NOTES.md`.

## Gate A update (2026-10-22): ready for review before the freeze

Work on `main`, merged from one branch per task: `env-pins`,
`a2-freeze-integrity`, `a1-harmonise`, `a3-models`, `a4-pdf-check`,
`gate-c-prep`, `gate-a-final`. Decisions are D24–D34 in `DECISIONS.md`.
**NFHS-6 has still not been scored.** The sections further down describe the
base pipeline of 2026-10-18 and are kept for reference. Where they differ,
the numbers here supersede them.

**Freeze artifact:** `outputs/models/frozen_models.pkl`, training-data hash
`d3f06bee428deeef` (OS-independent since D25), config hash `24e4b2befa146b0a`, 571 training
units, 684 test units, 547 stable units, 23 indicators, and 16 methods
(`manifest.json`).

### What changed

| task | result |
|---|---|
| A2 freeze integrity | The hash no longer depends on line endings. The NFHS-6 evaluation script runs only if tag `freeze-v1` == HEAD, the tree is clean, and it has never run before; it records the commit id (D25, D26). |
| A1 harmonisation | `harmonise.py` builds a lineage crosswalk with a source and confidence per edge, union-find components, population-weighted aggregation with SE propagation, and flagged inheritance with extra variance (D27–D31). Training: 571 units (+24). Test: 684 units (+137 = 121 districts changed between NFHS-4 and NFHS-5 but stable between NFHS-5 and NFHS-6, plus 16 boundary-changed remnants). 165 cases are listed for review. Official lineage could not be fetched (the LGD download needs a captcha), so none was invented. |
| A3 models | Fay-Herriot (REML) replaces EB; FH-RTM; national trend; spatial-neighbour features; ablations; a no-state-dummy variant; seen-state CV; paired state-clustered bootstrap tests with BH; secondary metrics. All pre-registered in D32 before they were run. |
| A4 PDF check | An independent PDF parser and a random spot-check script. **Not run: needs the IIPS PDFs** (D33, `docs/NFHS6_PDF_CHECK.md`). |
| Gate C prep | Conformal intervals (split, normalised, shift-weighted), BH-FDR surprises with a naive top-k comparison, per-district noise ceiling, forecastability decomposition and map. Pre-registered in D34 and checked in `--dev` mode on NFHS-4 → NFHS-5 (`outputs/gate_c_dev/`). Report skeleton in `docs/report.md`. |

### NFHS-4 → NFHS-5 LOSO results (harmonised training set, 571 districts)

Skill vs persistence with 95% state-clustered CI. OVERALL pools squared
errors; MEAN is the unweighted mean over the 23 indicators.

| method | OVERALL skill | MEAN skill | skill net of national shift (OVERALL) |
|---|---|---|---|
| national trend (= state trend in LOSO) | 0.422 [0.325, 0.478] | 0.337 | 0 (reference) |
| RTM | 0.613 [0.535, 0.672] | 0.540 | 0.330 [0.252, 0.396] |
| Fay-Herriot (denoised persistence) | 0.083 | 0.108 | −0.586 |
| FH-RTM | 0.624 [0.551, 0.678] | 0.554 | 0.350 |
| elastic net | 0.637 [0.566, 0.690] | 0.577 | 0.373 [0.308, 0.424] |
| LightGBM | 0.630 [0.552, 0.687] | 0.557 | 0.360 |
| elastic net, no state dummies | 0.625 | 0.565 | 0.352 |
| LightGBM, no state dummies | 0.627 | 0.556 | 0.355 |

On the stable subset (547 districts) the numbers are within 0.003: RTM
0.613, elastic net 0.639, LightGBM 0.630 (`results_nfhs4to5_stable.csv`).
Ablations (elastic net, OVERALL): own history 0.611, + other indicators
0.611, + spatial neighbours 0.632, + state effects (full) 0.637. The spatial
features carry the ML gain over RTM. Cross-indicator information adds
nothing.

**ML vs RTM, paired test (D32):**
- **Elastic net:** beats RTM pooled (ΔMSE −4.9, 95% CI [−7.7, −2.0], about
  6% lower MSE). Per indicator it is significantly better on only 2 of 23
  after BH (child marriage, teen pregnancy; on the stable subset 4: plus PNC
  and women overweight) and never significantly worse.
- **LightGBM and both no-state variants:** not significantly different from
  RTM overall.

So "ML only ties RTM" (base pipeline) becomes "elastic net with spatial
features is modestly but significantly better than RTM in pooled MSE on
4→5. The gain is spread thin across indicators."

**Seen-state CV (optimistic reference):** elastic net 0.763, RTM 0.637,
state trend 0.597. The large gain comes from state dummies learning the
same-transition state shocks, which a real forecast cannot know (bug 5.3).

**Noise (D32 secondary metrics):** sampling noise is 24% of the variance of
district change (pooled; 32% mean over indicators). The ceiling is 0.93
relative to persistence and **0.83 relative to RTM**, so the best method
recovers about 6% of the RTM error out of a possible 83%.

### Review files for the team

1. `outputs/crosswalk_review.csv` (165 rows): 132 districts changed between
   NFHS-4 and NFHS-5 (need Census-2011 parents and NFHS-4 values), 16 new
   NFHS-6 districts without a parent, 16 retained-name boundary changes
   (scored by inheritance), and 1 low-confidence inferred parent
   (Bilaspur → Gaurela-Pendra-Marwahi).
2. `outputs/harmonisation_report.md` and `outputs/harmonised_units.csv`.
3. `DECISIONS.md` D24–D34, especially the pre-registrations D32 and D34.
4. `outputs/paired_summary_nfhs4to5*.csv`, `outputs/secondary_nfhs4to5*.csv`.

### Gate B (humans only)

1. Review the files above. Optionally supply `data/lineage/*.csv` and the
   IIPS PDFs, then re-run `scripts/run_pipeline.py` and
   `scripts/spot_check_nfhs6.py` and commit. Any change to training data or
   config re-freezes the models.
2. Make sure the tree is clean (`git status`) and HEAD is the commit to freeze.
3. `git tag -a freeze-v1 -m "Models frozen before NFHS-6 evaluation"` and
   `git push origin main freeze-v1`.
4. `.venv\Scripts\python scripts\evaluate_nfhs6.py --confirm-frozen` (once;
   it refuses a second run).
5. Commit the NFHS-6 outputs and `outputs/nfhs6_evaluation_run.json`, then run
   `scripts\gate_c_analysis.py --predictions outputs\predictions_nfhs5to6.parquet`
   and fill `docs/report.md`.

---

## Base pipeline (2026-10-18)

Branch `base-pipeline`, built 2026-10-18. Models are trained and validated on
NFHS-4 → NFHS-5 and frozen. **NFHS-6 has not been scored**; that is the team's
next step, after review (see "How to run").

### What was built

- A Python package, `nfhs_forecast/`, that:
  - loads both sources and cleans them (suppressed cells, India.csv's zero
    encoding, name variants, source parse errors);
  - matches indicators across rounds by normalised text (never by
    `indicator_no`);
  - matches districts within state;
  - audits the two copies of NFHS-5 against each other;
  - builds a tidy panel and an indicator codebook;
  - attaches assumed standard errors;
  - runs four baselines and two ML models, evaluates them with
    leave-one-state-out (LOSO) CV and a state-clustered bootstrap, and
    freezes the models.
- `scripts/run_pipeline.py` runs everything with one command (~3 min) and
  never scores NFHS-6.
- `scripts/evaluate_nfhs6.py` is the one-shot out-of-time test. It is written
  and tested on synthetic data but **not run**.
- 74 pytest tests, all passing. They cover value cleaning, name and indicator
  normalisation, indicator and district matching, the SE function, each
  baseline, the no-leakage asserts, and a synthetic end-to-end run of the
  NFHS-6 script.
- Docs: `README.md`, `DECISIONS.md` (D01–D23), `docs/DATA_SOURCES.md`,
  `outputs/data_report.md`.
- Stubs with TODOs: `harmonise.py` (boundary crosswalk), `parse_nfhs6.py`
  (independent PDF parser).

### How to run

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\run_pipeline.py      # whole pipeline
.venv\Scripts\python -m pytest -q                 # tests
# final test, only after review + tag:
git tag -a freeze-v1 -m "Models frozen before NFHS-6 evaluation"
.venv\Scripts\python scripts\evaluate_nfhs6.py --confirm-frozen
```

The frozen models are `outputs/models/frozen_models.pkl` (training-data hash
`4f73a5ed88b1143e` since the line-ending fix in D25; identical across reruns and operating systems). `evaluate_nfhs6.py` aborts if the
NFHS-4/5 data no longer match that hash.

### Files moved and deleted

| From | To |
|---|---|
| `India.csv` | `data/raw/nfhs4_5/India.csv` |
| `Districts.geojson` | `data/raw/geo/Districts.geojson` |
| `nfhs6-district-data-india-main/data/*.csv` (36 files) | `data/raw/nfhs6/` |
| `nfhs6-district-data-india-main/{CITATION.cff, LICENSE, README.md}` | `data/raw/nfhs6/SOURCE/` |
| `CodeFlow_Jackfruit_OnePager.docx` | `docs/` |
| `nfhs6-district-data-india-main/scripts/` and the emptied folder | **deleted** |
| `README.md` | overwritten with the project README |

No raw file exceeds 50 MB (largest: `Districts.geojson`, 28 MB), so all raw
data are tracked.

### Districts and indicators

| | count |
|---|---|
| India.csv districts (after cleaning) | 707 |
| NFHS-6 districts (after excluding the merged DNH&DD block) | 711 |
| NFHS-6 districts matched to India.csv | 694 (633 exact, 6 ignoring spaces, 55 via alias table, 0 fuzzy) |
| NFHS-6 districts unmatched | 17, all new districts flagged `boundary_change_single_round` |
| **Core districts used for modelling** | **547 in 30 states/UTs** |
| excluded: no NFHS-4 values (boundary changed 4→5) | 127 |
| excluded: NFHS-6 boundary flag | 16 matched + 17 unmatched |
| excluded: not in NFHS-6 (Manipur ×9, Puducherry district, DNH, Daman, Diu) | 13 |
| excluded: NFHS-5 audit failure (Maharajganj, Mahoba, Chandigarh, Lakshadweep) | 4 |
| cross-round indicator matches (exact / prefix / fuzzy) | 72 (67 / 4 / 1) |
| comparable in all three rounds | 46 |
| **included (core set)** | **23** |

The 23 included indicators are listed by domain below:

| domain | indicators |
|---|---|
| WASH | water_improved, menstrual_hygiene |
| women's empowerment / FP | women_10yr_school, child_marriage, teen_pregnancy, fp_modern, unmet_need |
| maternal care | anc_first_trimester, anc_4plus, ifa_100d, pnc_mother, inst_births, csection |
| child health / vaccination | full_vacc, mcv1, vitamin_a, diarrhoea, early_bf |
| nutrition | stunting, wasting, underweight, women_thin, women_overweight |

The other 23 comparable indicators (e.g. BCG, polio, skilled birth attendance,
female sterilisation, electricity, iodised salt, severe wasting) have
`include = False` in `data/processed/indicator_codebook.csv`. Each can be
switched on by adding it to `indicators.core` in `config.yaml`.

### Data audit (NFHS-6 file's own NFHS-5 column vs India.csv NFHS-5)

- 49,967 matched district × indicator pairs; 46,395 have a number in both.
- **Exact agreement: 95.3%. Within 0.5 points: 96.8%.**
- 2,386 of the 2,394 `*` cells in the NFHS-6 file are `0` in India.csv. This
  is the evidence that India.csv zeros encode suppression (D08).
- **Systematic disagreements:**
  - `health_insurance` agrees in only 52% of districts. The NFHS-6 sheets
    restate NFHS-5 higher in some states, so it was removed from the core
    set.
  - Children's second MCV dose agrees in 0.3% (not used).
- Per-district: **Maharajganj and Mahoba (UP) are swapped** between the
  sources. Chandigarh and Lakshadweep have misaligned India.csv rows. Delhi's
  districts disagree too, but they were already out (no NFHS-4).

### Results: NFHS-4 → NFHS-5, leave-one-state-out CV

547 districts, 23 indicators, 30 folds (one per state/UT). Skill is
1 − MSE(method)/MSE(persistence). The OVERALL row pools squared errors over
indicators. CIs are 95%, from a bootstrap over states.

| method | MAE [95% CI] | RMSE | skill [95% CI] |
|---|---|---|---|
| Persistence | 10.15 [9.26, 10.92] | 14.14 | 0 (reference) |
| State trend (national trend in LOSO) | 7.50 [7.12, 7.96] | 10.75 | 0.422 [0.322, 0.480] |
| Regression to the mean | 6.25 [6.00, 6.56] | 8.80 | 0.613 [0.529, 0.664] |
| EB shrinkage | 9.81 [8.90, 10.59] | 13.53 | 0.085 [0.073, 0.101] |
| Elastic net | 6.20 [5.95, 6.47] | 8.79 | 0.614 [0.529, 0.667] |
| LightGBM | 6.24 [6.00, 6.49] | 8.80 | 0.613 [0.529, 0.667] |
| State trend, oracle (reference only) | 6.32 [6.06, 6.63] | 8.98 | 0.597 [0.518, 0.644] |

Per indicator (skill vs persistence; "ceiling" = the skill a perfect
forecaster of the true value would reach, given the assumed SEs):

| indicator | domain | persistence RMSE | state trend | RTM | EB | elastic net | LightGBM | oracle | ceiling |
|---|---|---|---|---|---|---|---|---|---|
| diarrhoea | child_health | 6.50 | 0.06 | 0.58 | 0.25 | 0.62 | 0.51 | 0.37 | 0.91 |
| early_bf | child_health | 16.10 | -0.03 | 0.15 | 0.08 | 0.17 | 0.16 | 0.29 | 0.94 |
| full_vacc | child_health | 23.00 | 0.55 | 0.75 | 0.14 | 0.73 | 0.73 | 0.63 | 0.92 |
| mcv1 | child_health | 13.90 | 0.39 | 0.74 | 0.30 | 0.71 | 0.68 | 0.49 | 0.86 |
| vitamin_a | child_health | 18.50 | 0.12 | 0.52 | 0.18 | 0.58 | 0.48 | 0.57 | 0.85 |
| anc_4plus | maternal_care | 19.00 | 0.20 | 0.46 | 0.05 | 0.42 | 0.49 | 0.47 | 0.96 |
| anc_first_trimester | maternal_care | 19.30 | 0.38 | 0.66 | 0.06 | 0.65 | 0.64 | 0.55 | 0.96 |
| csection | maternal_care | 7.90 | 0.45 | 0.45 | -0.05 | 0.46 | 0.49 | 0.54 | 0.81 |
| ifa_100d | maternal_care | 20.70 | 0.50 | 0.50 | 0.03 | 0.46 | 0.50 | 0.65 | 0.96 |
| inst_births | maternal_care | 13.00 | 0.65 | 0.79 | 0.11 | 0.78 | 0.80 | 0.72 | 0.96 |
| pnc_mother | maternal_care | 21.80 | 0.64 | 0.74 | 0.05 | 0.78 | 0.78 | 0.76 | 0.98 |
| stunting | nutrition | 7.70 | 0.12 | 0.38 | 0.14 | 0.40 | 0.38 | 0.26 | 0.76 |
| underweight | nutrition | 8.60 | 0.12 | 0.40 | 0.13 | 0.47 | 0.44 | 0.41 | 0.82 |
| wasting | nutrition | 8.70 | 0.04 | 0.47 | 0.22 | 0.54 | 0.47 | 0.26 | 0.87 |
| women_overweight | nutrition | 7.00 | 0.36 | 0.34 | 0.01 | 0.37 | 0.39 | 0.55 | 0.93 |
| women_thin | nutrition | 6.50 | 0.48 | 0.62 | 0.08 | 0.63 | 0.63 | 0.58 | 0.93 |
| menstrual_hygiene | wash | 23.70 | 0.77 | 0.87 | 0.03 | 0.86 | 0.86 | 0.87 | 0.97 |
| water_improved | wash | 5.80 | 0.20 | 0.47 | 0.09 | 0.48 | 0.44 | 0.26 | 0.97 |
| child_marriage | women_empowerment | 7.80 | 0.31 | 0.37 | 0.15 | 0.47 | 0.40 | 0.46 | 0.76 |
| fp_modern | women_empowerment | 17.00 | 0.32 | 0.59 | 0.02 | 0.58 | 0.64 | 0.53 | 0.98 |
| teen_pregnancy | women_empowerment | 3.80 | 0.14 | 0.28 | 0.28 | 0.32 | 0.22 | 0.20 | 0.63 |
| unmet_need | women_empowerment | 6.90 | 0.38 | 0.63 | 0.08 | 0.61 | 0.55 | 0.55 | 0.96 |
| women_10yr_school | women_empowerment | 8.20 | 0.61 | 0.62 | -0.00 | 0.65 | 0.66 | 0.68 | 0.93 |

**What this says (on NFHS-4 → NFHS-5 only):**
- **Persistence is a weak baseline here.** NFHS-4 → NFHS-5 saw large
  systematic changes (e.g. institutional births, vaccination, menstrual
  hygiene), so anything that captures the average change beats it.
- **Regression to the mean is the baseline to beat, and ML only ties it.**
  Elastic net and LightGBM match RTM overall (skill 0.61 for all three, with
  nearly identical CIs). ML is slightly ahead on 18 of 23 indicators, by at
  most about 0.1 skill (e.g. child marriage, wasting, underweight). No paired
  test has been run yet, so treat those gaps as unproven.
- **EB shrinkage alone gains little (0.08),** as expected: it denoises the
  starting value but adds no trend. Its biggest gains are on the noisiest
  small-denominator indicators (MCV1 0.30, teen pregnancy 0.28, diarrhoea
  0.25, wasting 0.22).
- Knowing each state's own change (oracle, 0.60) is no better than RTM/ML.
  Much of the predictable signal is national change plus regression toward
  the mean.
- The hardest indicators are early breastfeeding (≤ 0.17 for every honest method),
  teen pregnancy and the anthropometric indicators (stunting, underweight;
  ≤ 0.47).
- Skill ceilings under the assumed noise are 0.63–0.98, so most indicators
  are far from their noise limit. Whether that holds for NFHS-6
  (COVID-period fieldwork in NFHS-5) is exactly what the frozen test will
  show.

### Assumptions (all in `config.yaml`; reasons in `DECISIONS.md`)

1. **Sample sizes and design effect (D16).** Per-district n by group:
   households 900, women 1000, women_subgroup 200, mothers 250, men 140,
   children 300, children_subgroup 70. Design effect 2.0 (sensitivity
   1.5–3.0: EB skill 0.069–0.106, ceiling 0.95–0.90;
   `outputs/sensitivity_deff.csv`). Small-sample NFHS-6 rows use n = 37.
   These are not published figures.
2. India.csv zero handling thresholds: 80% / 95% / 1% (D08).
3. Audit thresholds: indicators need ≥ 90% and districts ≥ 95% NFHS-5
   agreement (D12).
4. The core indicator set is a judgement within the comparable set (D14).
5. Definition exclusions: pentavalent vs penta/DPT; adequate diet (D15).
6. The 55 district aliases (D11).
7. The state → region grouping used as a model feature (D18).
8. Model settings:
   - LightGBM is fixed and small, not tuned;
   - elastic net alpha and l1_ratio are tuned in inner CV grouped by state;
   - models predict the change from the previous round (D18).
9. State trend extrapolates the NFHS-4→5 state change to NFHS-5→6 with
   damping 1.0 (D17).

### Problems found in the data

**India.csv (NFHS-4/5)**
- `na` and `*` are written as `0` (132 district-rounds and 14,950 values are
  "not available / not measured"; 5,518 zeros were treated as suppressed).
- `Saraikela-Kharsawan` is parsed under the name `Sahibganj`.
- A placeholder district `Jammu & Kashmir | Data Not Available`.
- A conflicting duplicate row: Surajpur, men's alcohol use.
- Trailing spaces in state names.
- Scrambled single rows in Meghalaya's code columns.
- Text lost its hyphens (`2024` for `20-24`).
- Chandigarh and Lakshadweep have misaligned rows (implausible teen
  pregnancy, child marriage and menstrual hygiene values).
- Dhule's tobacco/alcohol rows are shifted by one.
- The children's second-MCV-dose values disagree with the NFHS-6 file in
  almost every district.

**NFHS-6 file**
- `*` suppressed cells (3,774).
- `small_sample_flag` is per row, not per round.
- Chandigarh and Lakshadweep come from the state sheet (different numbering,
  101 rows).
- `Dadra & Nagar Haveli And Daman & Diu` merges several districts (380 rows).
- Indicator text is truncated in some states, with footnote-spacing variants.
- Categories are carried over wrongly (women's indicators under `Men`).
- Puducherry district itself is missing.

**Between the sources**
- Maharajganj and Mahoba (UP) have swapped values.
- Health-insurance NFHS-5 values differ in about half of the districts.
- Delhi's districts do not agree.

Each of these should be checked against the IIPS PDFs.

### Next steps

1. **Review and freeze.** Read `DECISIONS.md` and the codebook, agree the core
   indicator set, tag the commit, then run `scripts/evaluate_nfhs6.py
   --confirm-frozen` once. Any change made after seeing NFHS-6 results must be
   logged.
2. **NFHS-6 evaluation write-up:** skill per indicator under shift, and a
   comparison with the NFHS-4→5 CV results above.
3. **Boundary harmonisation** (`harmonise.py`): a parent-child crosswalk to
   NFHS-5 boundaries with population weights and SE propagation. This would
   bring back about 160 districts.
4. **Fay-Herriot** small-area model (statsmodels is installed) to replace the
   simple EB, ideally with covariates.
5. **Conformal prediction intervals,** calibrated on NFHS-4→5 and tested for
   coverage on NFHS-5→6.
6. **FDR-controlled surprise detection:** districts whose NFHS-6 value is far
   from the forecast relative to noise, at a controlled false-discovery rate.
7. **Aspirational Districts (ADP)** descriptive comparison (list of 112
   districts needed).
8. **Smaller items:**
   - a paired bootstrap test for ML vs RTM;
   - verify the provisional NFHS-6 values and the swaps above against the
     IIPS PDFs (`parse_nfhs6.py`);
   - optionally score districts that lack NFHS-4 on NFHS-5→6.
