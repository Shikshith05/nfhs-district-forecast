# CLAUDE.md — nfhs-district-forecast (Team CodeFlow, "Jackfruit Problem")

Read this first. It says what exists, what is broken, and what to build next.
Also read `README.md`, `DECISIONS.md` (D01–D23) and `outputs/SUMMARY.md`.

## 1. Project in one paragraph

Title: *How Predictable Are District Health Outcomes? A Noise-Aware, Out-of-Time
Test of Forecasting and Programme Surprise on India's NFHS-6.*
We train forecasting models on the NFHS-4 → NFHS-5 transition, FREEZE them,
predict NFHS-6 (2023-24) from NFHS-5 ONCE, and score against the released NFHS-6
district values. Each district estimate is noisy (small samples), so we also
measure a noise ceiling, check conformal coverage under shift, and flag only
"surprise" districts that survive false-discovery-rate control.
Central claim: forecastability is a property of an indicator AND a district,
not only of a model. Graded on quantity, quality, novelty.
Mentor feedback: "Excellent forecasting setup, but your entire project hinges on
a mathematically sound parent-child mapping for district boundary changes."

## 2. HARD RULES (never break)

1. NFHS-6 outcome values are NEVER used for training, tuning, feature
   engineering, model selection, indicator/district selection, or choosing
   harmonisation weights. They are read only for loading/cleaning/audit and the
   final one-shot score. Training code calls `models.assert_no_round6()`; keep it.
2. NEVER run `scripts/evaluate_nfhs6.py`. That is the one-shot test; a human
   reviews, tags `freeze-v1`, and runs it. Do not read NFHS-6 outcomes to "check"
   anything either.
3. Any change after the freeze must be logged in `DECISIONS.md`.
4. Log every judgement call in `DECISIONS.md` (next number is D24).
5. Run `python -m pytest -q` after every change; keep all tests passing and add
   tests for new code. Put tunables in `config.yaml`, not in code.
6. Small commits, one branch per task. Do not touch `data/raw/`.
7. Windows PowerShell, Python 3.11, `.venv`. Keep code OS-independent.

## 3. What is implemented (verified: 74 tests pass, results reproduce)

Package `nfhs_forecast/`: `config.py`, `load.py`, `clean.py` (value/name/indicator
normalisation; India.csv zeros reclassified as suppressed), `match.py`
(indicator matching by text; district matching within state; NFHS-5 audit of the
two copies; panel + codebook), `noise.py` (assumed SEs), `baselines.py`
(persistence, state trend, regression-to-mean, EB shrinkage), `models.py`
(features, elastic net, LightGBM, leakage assert, frozen artifact),
`evaluate.py` (leave-one-state-out CV, metrics, state-clustered bootstrap, plot),
`report.py`. `scripts/run_pipeline.py` runs everything (~3 min) and never scores
NFHS-6. `scripts/evaluate_nfhs6.py` is written but NOT run.

Data: NFHS-4/5 from `India.csv` (community parse of IIPS fact sheets); NFHS-6
from a community CSV compilation (provisional). Panel key:
`district_uid = "<state>|<district>"`; columns include `indicator_id`, `round`,
`value`, `se`, `small_sample_flag`. Processed outputs go to `data/processed/`
(git-ignored), e.g. `panel_model.parquet`, `district_name_map.csv`,
`matched_districts.csv`, `indicator_codebook.csv`.

Numbers: 547 core districts (of ~707) in 30 states/UTs; 23 indicators (of 46
comparable); NFHS-5 audit agreement 95.3% exact. LOSO skill vs persistence
(pooled): state trend 0.42, RTM 0.613, elastic net 0.614, LightGBM 0.613,
EB shrinkage 0.08. ML only ties RTM; no paired test run yet. Noise assumptions
(per-group n, design effect 2.0, sensitivity 1.5–3) are ASSUMED, not published.

## 4. What is NOT implemented (stubs / missing)

- `harmonise.py`: STUB (raises NotImplementedError). Biggest gap.
- `parse_nfhs6.py`: STUB (independent PDF parser for spot-checking NFHS-6).
- Fay-Herriot, spatial-neighbour features, feature ablations, paired tests.
- Conformal intervals, shift tests, FDR surprise detection.
- Per-district noise ceiling / forecastability map.
- Aspirational-districts analysis (RQ5), SHAP/clustering, dashboard: OUT OF SCOPE
  for the 75% target (list as future work).
- Models: only 2 ML families (one-pager promises more; keep wording honest).

## 5. Known bugs / weaknesses — fix or report

1. **Freeze hash depends on OS.** `models.training_data_hash` uses
   `to_csv(...)` with the default line terminator, so Windows (CRLF) and
   Linux/Mac (LF) hash differently (verified: committed `791f980116a74d5a` is
   the CRLF hash; LF gives `4f73a5ed88b1143e` on identical data). Fix: pass
   `lineterminator="\n"`, re-run the pipeline, update the manifest/SUMMARY hash.
2. **"Run once" is not enforced.** `evaluate_nfhs6.py` only needs
   `--confirm-frozen`. Add checks: git tag `freeze-v1` exists and points at
   HEAD, working tree clean, refuse if `nfhs6_evaluation_run.json` already
   exists, record the commit id in the run log.
3. **CV does not match the real test.** LOSO validates on unseen states, so the
   state one-hot features are never exercised; the NFHS-6 test uses seen states.
   Also only one training transition exists: most skill is the national mean
   shift (state-trend/national-only already gets 0.42 of 0.61). Pre-register
   extra variants BEFORE the run (e.g. ML without state dummies); report skill
   net of the national shift as a secondary metric.
4. **State trend** is validated as a national trend (LOSO) but scored as a
   state-specific extrapolation on NFHS-6. Say so, or validate both.
5. **Noise ceiling** is measured against persistence (weak here) and rests on
   assumed n/deff, so it looks far above achieved skill (mean 0.90 vs 0.61).
   Also report noise as a share of the variance of change and the ceiling
   relative to RTM.
6. **Pooled OVERALL skill** is dominated by high-variance indicators. Also
   report the unweighted mean over indicators.
7. **ML vs RTM claim** in SUMMARY ("18 of 23") is not reproduced: elastic net
   beats RTM on 14, LightGBM on 11, best-of-two on 17. Run the paired
   state-clustered bootstrap test before claiming anything.
8. **Selection:** the 127 districts dropped for no NFHS-4 values are boundary-
   changed districts, not random. State this in the report.

## 6. Task list to reach ~75% (in dependency order)

**Gate A — before the freeze (training data may change here)**
- A1. Implement `harmonise.py`. Build a lineage crosswalk to NFHS-5 boundaries
  (unchanged / renamed / split / merged / multi-parent). Use a bipartite graph
  between NFHS-5 and NFHS-6 districts; each connected component is one analysis
  unit. Inputs available: Census-2011 codes in India.csv (`district_code`; ~65
  newer districts lack codes), `Districts.geojson` (Census 2011 polygons), the
  NFHS-6 `boundary_change_single_round` flag, `has_nfhs4` in
  `matched_districts.csv`. Weights come from Census/LGD population, NEVER from
  NFHS-6 outcomes. Combine children into parents with weights; SE_parent =
  sqrt(sum w_i^2 se_i^2) plus extra variance where weights are estimated; flag
  harmonised rows; NFHS-4 values for districts split between 2015 and 2019 are
  inherited from the parent, flagged, extra variance. Acceptance: tests
  (weights sum to 1; a re-aggregated split reproduces its parent on synthetic
  data), published state values rebuilt within a pre-set tolerance, results
  repeated on the stable-boundary subset, bootstrap clustered by parent district.
  Expected to recover ~160 districts. Log in DECISIONS.md.
- A2. Fix bugs 5.1 and 5.2. Re-run pipeline; review outputs.
- A3. Models: add Fay-Herriot (statsmodels) to replace simple EB; spatial-
  neighbour features; feature-set ablations (own history / + cross-indicator /
  + spatial / + state effects); paired state-clustered bootstrap tests ML vs RTM.
- A4. Finish a random-sample spot-check of NFHS-6 values against the official
  IIPS PDFs (`parse_nfhs6.py`); report the agreement rate.

**Gate B — human only:** review, re-freeze, tag `freeze-v1`, run
`evaluate_nfhs6.py --confirm-frozen` once.

**Gate C — after the run**
- C1. Conformal intervals (split + one shift-weighted/adaptive variant),
  calibrated on NFHS-4→5, coverage measured on NFHS-6. Develop early on LOSO
  residuals.
- C2. FDR (Benjamini–Hochberg) surprise detection on residuals scaled by forecast
  plus sampling variance; compare with naive top-k.
- C3. Per-district noise ceiling and forecastability map.
- C4. Released harmonised panel + codebook (CSV); final report with figures.

## 7. Style

Match existing code: type hints, docstrings that state the leakage contract,
config-driven, tests in `tests/`. Explain statistical choices in DECISIONS.md.
Keep claims honest: state assumptions (SEs, weights) and report null results
(e.g. "ML does not beat RTM") as findings.
