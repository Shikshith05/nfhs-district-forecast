# nfhs-district-forecast

**Forecasting district health outcomes in India: a noise-aware out-of-time test on NFHS-6.**
Team CodeFlow (Saurav S, Shikshith V, Thijl Thejaswi MN, Vishwas V Angadi).

India's National Family Health Survey (NFHS) reports district-level health
indicators every 3–4 years: NFHS-4 (2015-16), NFHS-5 (2019-21) and NFHS-6
(2023-24). Each district estimate rests on a small sample and is noisy. This
project:

1. **trains** forecasting models on the NFHS-4 → NFHS-5 transition only;
2. **freezes** them;
3. **predicts** NFHS-6 from NFHS-5 and **tests** against the real NFHS-6 values.

ML models (elastic net, LightGBM) are compared with simple baselines
(persistence, state trend, regression to the mean, empirical-Bayes shrinkage),
taking assumed sampling noise into account. See
`docs/CodeFlow_Jackfruit_OnePager.docx` for the full proposal.

> **Status:** base pipeline. Models are trained and validated on NFHS-4 → NFHS-5
> and frozen. **NFHS-6 has not been scored yet**: the team reviews, tags the
> freeze, then runs `scripts/evaluate_nfhs6.py` once. Results so far:
> `outputs/SUMMARY.md`.

## Setup (Windows, PowerShell)

Requires Python 3.11 (`py --list` shows what is installed).

```powershell
git clone https://github.com/Shikshith05/nfhs-district-forecast.git
cd nfhs-district-forecast
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks `activate`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation
and call `.venv\Scripts\python` directly as below.

## Run

```powershell
# whole pipeline: clean -> match -> audit -> report -> CV on NFHS-4->5 -> freeze (~3 min)
.venv\Scripts\python scripts\run_pipeline.py

# data stages only (seconds)
.venv\Scripts\python scripts\run_pipeline.py --skip-cv

# tests (~10 s)
.venv\Scripts\python -m pytest -q
```

**The final test. Do not run it until the freeze is reviewed and tagged:**

```powershell
git tag -a freeze-v1 -m "Models frozen before NFHS-6 evaluation"
git push origin freeze-v1
.venv\Scripts\python scripts\evaluate_nfhs6.py --confirm-frozen
```

It refuses to run without `--confirm-frozen`, and aborts if the NFHS-4/5
training data changed since the freeze.

## No-leakage rule

NFHS-6 values are never used for training, tuning, feature engineering or
model selection. They appear only in loading, cleaning, the data report and
the NFHS-5 audit. Training code calls `models.assert_no_round6()` and fails if a
round-6 row reaches it; tests enforce this. Indicator and district selection
use NFHS-4/5 values and NFHS-6 *metadata* (which districts/indicators exist,
boundary flags) only.

## Assumptions you should know about

The fact sheets do not publish sample sizes, so **standard errors are
assumptions**: SE = sqrt(p(1−p)/n · deff), with assumed per-district n by
indicator group and design effect 2.0 (sensitivity range 1.5–3). They are all
in `config.yaml` under `noise:` and explained in `DECISIONS.md` (D16). Rows
flagged small-sample in NFHS-6 get n = 37. No district boundary harmonisation
is done yet: only districts that match cleanly across all three rounds are
used (547 of ~707).

## Repository structure

```
nfhs_forecast/            Python package
  config.py               load config.yaml, paths, hashes
  load.py                 raw readers (India.csv, NFHS-6 CSV, GeoJSON properties)
  clean.py                value / name / indicator-text normalisation, per-source cleaning
  match.py                indicator + district matching, panel, NFHS-5 audit, codebook
  noise.py                assumed standard errors
  baselines.py            persistence, state trend, regression to the mean, EB shrinkage
  models.py               features, elastic net, LightGBM, no-leakage assert, frozen artifact
  evaluate.py             leave-one-state-out CV, metrics, state-clustered bootstrap, plot
  report.py               outputs/data_report.md
  harmonise.py            STUB: boundary crosswalk (TODO)
  parse_nfhs6.py          STUB: independent NFHS-6 PDF parser (TODO)
scripts/
  run_pipeline.py         one-command pipeline (never scores NFHS-6)
  evaluate_nfhs6.py       one-shot NFHS-6 test with frozen models (NOT YET RUN)
tests/                    pytest suite; tests/fixtures/ holds small synthetic CSVs
data/raw/                 tracked raw data: nfhs4_5/, nfhs6/ (+ SOURCE/ attribution), geo/
data/processed/           generated (git-ignored): panel, codebook, name maps, audit
outputs/                  data_report.md, results_nfhs4to5.csv, skill_by_indicator.png,
                          sensitivity_deff.csv, models/ (frozen models), SUMMARY.md
docs/                     one-pager, DATA_SOURCES.md
config.yaml               all tunables and assumptions
DECISIONS.md              every judgement call, with reasons
```

## Key outputs

| File | What |
|---|---|
| `outputs/SUMMARY.md` | what was built, counts, results, assumptions, data problems, next steps |
| `outputs/data_report.md` | rows / districts / indicators per round, missingness, exclusions, audit |
| `outputs/results_nfhs4to5.csv` | MAE, RMSE, skill vs persistence (95% CI) per indicator and method |
| `outputs/skill_by_indicator.png` | skill by indicator, one panel per method |
| `data/processed/indicator_codebook.csv` | every indicator: texts per round, match, coverage, include flag + reason |
| `data/processed/indicator_variants.csv` | every raw indicator text variant and how it was matched (review this) |
| `data/processed/district_name_map.csv` | every district match decision with score, audit agreement, status |
| `data/processed/panel_long.parquet` | tidy panel: district × indicator × round, all districts |

`data/processed/` is git-ignored; run the pipeline to create it.

## Data sources

- **NFHS-4/5:** `India.csv` from https://github.com/SaiSiddhardhaKalla/NFHS, parsed
  from the IIPS NFHS-5 district fact sheets (with NFHS-4 comparison column),
  Census 2011 codes (about 65 newer districts lack codes).
- **NFHS-6:** Patra, N. K. (2026). *NFHS-6 Provisional District-Level Data
  (India): Independent Compilation.* GitHub. Parsed from the IIPS provisional
  State & District Fact Sheets. Attribution files: `data/raw/nfhs6/SOURCE/`.
- **Original publisher:** International Institute for Population Sciences (IIPS), Mumbai.

Details, licences and citations: `docs/DATA_SOURCES.md`.
