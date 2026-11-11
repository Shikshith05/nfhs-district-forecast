# nfhs-district-forecast

**Forecasting district health outcomes in India: a noise-aware out-of-time test on NFHS-6.**
Team CodeFlow (Saurav S, Shikshith V, Thijl Thejaswi MN, Vishwas V Angadi).

India's National Family Health Survey (NFHS) reports district-level health
indicators every 3–4 years: NFHS-4 (2015-16), NFHS-5 (2019-21) and NFHS-6
(2023-24). Each district estimate rests on a small sample and is noisy. This
project:

1. **trains** forecasting models on the NFHS-4 → NFHS-5 transition only;
2. **freezes** them;
3. **predicts** NFHS-6 from NFHS-5 once and **tests** against the real NFHS-6 values;
4. measures a noise ceiling, checks conformal coverage under shift, and flags only
   "surprise" districts that survive false-discovery-rate control.

Methods: persistence, national and state trends, regression to the mean (RTM),
Fay–Herriot, FH + RTM, elastic net and LightGBM (with ablations), all compared
with paired state-clustered bootstrap tests. See `docs/CodeFlow_Jackfruit_OnePager.docx`
for the proposal and `docs/report.md` for the report.

> **Status (2026-10-23): ready for Gate B.** Models are trained on NFHS-4 → NFHS-5 and
> frozen (training-data hash `d3f06bee428deeef`). **NFHS-6 has not been scored yet.**
> Everything after the one-shot test is automated: one command produces all tables,
> figures and the filled-in report. Read `docs/HUMAN_CHECKLIST.md` before Gate B.
> Results so far: `outputs/SUMMARY.md`. Report (NFHS-6 slots pending): `docs/report.md`.

## Setup (Windows, PowerShell)

Requires Python 3.11 (`py --list` shows what is installed).

```powershell
git clone https://github.com/Shikshith05/nfhs-district-forecast.git
cd nfhs-district-forecast
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

If Windows Smart App Control blocks a package DLL, see DECISIONS D24 (the pins in
`requirements.txt` load on the team machine). The code is OS-independent; on Linux or
macOS use `.venv/bin/python` and forward slashes.

## Reproduce everything, in order

| Step | Command | What it does |
|---|---|---|
| 1 | `.venv\Scripts\python scripts\run_pipeline.py --data-only` | rebuilds `data/processed/` (git-ignored) and the harmonisation outputs from `data/raw/`, and prints the training-data hash, which must equal `outputs/models/manifest.json`. **No training, no re-freeze.** |
| 2 | `.venv\Scripts\python -m pytest -q` | the test suite (about 90 s), including a full stand-in run of the post-run pipeline |
| 3 | `.venv\Scripts\python scripts\preflight.py` | PASS / WARN / FAIL: clean tree, tests, leakage tests, hash reproduces, crosswalk vs test districts, pre-registration order |
| 4 | `powershell -ExecutionPolicy Bypass -File scripts\gate_b.ps1` | **Gate B, humans only, once.** Pre-flight → type YES → tag `freeze-v1` → the one-shot NFHS-6 evaluation → commit its outputs. Stops on any error and never retries. |
| 5 | `.venv\Scripts\python scripts\post_run.py` | after Gate B: every post-run table and figure (`outputs/post_run/`), `docs/report.md` filled in and number-checked, release with NFHS-6 |

Other commands:

```powershell
# Full retraining + CV + freeze (~7-10 min on 8 cores). This RE-FREEZES the models:
# only before Gate B, and only if training data or methods changed (log it in DECISIONS.md).
.venv\Scripts\python scripts\run_pipeline.py

# Stand-in run of step 5 on the NFHS-4 -> NFHS-5 predictions. Writes outputs/dryrun/ only;
# every file is named and marked DRYRUN and is NOT an NFHS-6 result.
.venv\Scripts\python scripts\post_run.py --dryrun

# Re-render the report (before Gate B the NFHS-6 slots stay pending) and check every number.
.venv\Scripts\python scripts\build_report.py
.venv\Scripts\python scripts\check_report_numbers.py

# Release tables (NFHS-4/5 only before Gate B).
.venv\Scripts\python scripts\make_release.py

# NFHS-6 data check against the IIPS PDFs (PDFs + manifest in data/raw/pdfs/; docs/NFHS6_PDF_CHECK.md).
.venv\Scripts\python scripts\spot_check_nfhs6.py
```

The one-shot evaluation refuses to run unless `--confirm-frozen` is given, the tag
`freeze-v1` points at HEAD, the working tree is clean, the NFHS-4/5 training data match
the frozen hash, and `outputs/nfhs6_evaluation_run.json` does not exist yet (it runs
once; D26). Use `scripts/gate_b.ps1` rather than calling it directly.

## No-leakage rule

NFHS-6 values are never used for training, tuning, feature engineering, model or
indicator selection, or harmonisation weights. Before Gate B they appear only in
loading, cleaning, the data report and the NFHS-5 audit. Training code calls
`models.assert_no_round6()` and fails if a round-6 row reaches it; tests enforce this.
Indicator and district selection use NFHS-4/5 values and NFHS-6 *metadata* (which
districts and indicators exist, boundary flags) only. The release holds NFHS-4/5 only
until the one-shot test has run.

## Assumptions you should know about

- **Standard errors are assumptions.** The fact sheets do not publish sample sizes:
  SE = sqrt(p(1−p)/n · deff), with assumed per-district n by population group and design
  effect 2.0 (sensitivity 1.5–3). See `config.yaml` → `noise` and DECISIONS D16.
- **Harmonisation is partial.** It uses only lineage the repository supports (D27–D31):
  571 training units, 684 NFHS-6 test units, and a 547-district stable-boundary subset on
  which every result is also reported. Inherited values carry an assumed extra variance.
  Official lineage, populations and NFHS-4 Census-2011 values can be supplied in
  `data/lineage/`.
- **NFHS-6 data are provisional** and parsed from PDFs by a third party. The PDF
  spot-check is built but needs the official PDFs (`docs/NFHS6_PDF_CHECK.md`).

## Repository structure

```
nfhs_forecast/            Python package
  config.py               config.yaml, paths, hashes, OS-independent writers
  load.py, clean.py       raw readers; value / name / indicator-text normalisation
  match.py                indicator + district matching, panel, NFHS-5 audit, codebook
  noise.py                assumed standard errors
  harmonise.py            boundary lineage crosswalk, analysis units, weighted aggregation
  spatial.py              Census-2011 polygon contiguity, neighbour features
  baselines.py            persistence, trends, RTM, EB, Fay-Herriot
  models.py               features, elastic net, LightGBM, no-leakage assert, frozen artifact
  evaluate.py             CV, metrics, paired tests, secondary metrics, skill plot
  freeze_guard.py         tag / clean-tree / run-once checks for the NFHS-6 test
  conformal.py, surprise.py, forecastability.py   Gate C analyses
  post_run.py             all post-run tables, figures and key numbers (DRYRUN-aware)
  report_build.py, report_check.py                report from template; number checker
  release.py              public CSV release
  parse_nfhs6.py          independent parser for the IIPS NFHS-6 PDFs
scripts/
  run_pipeline.py         data -> harmonisation -> CV -> freeze (never scores NFHS-6); --data-only
  preflight.py            checks before Gate B
  gate_b.ps1              Gate B helper (humans only)
  evaluate_nfhs6.py       one-shot NFHS-6 test with frozen models (run only via gate_b.ps1)
  post_run.py             everything after Gate B; --dryrun for the stand-in test
  build_report.py, check_report_numbers.py, make_release.py, spot_check_nfhs6.py
  gate_c_analysis.py      Gate C analyses on their own (D34; post_run.py includes them)
tests/                    pytest suite; tests/fixtures/ holds small synthetic CSVs
data/raw/                 tracked raw data: nfhs4_5/, nfhs6/ (+ SOURCE/ attribution), geo/
data/lineage/             optional supplied lineage / population / NFHS-4 Census-2011 files
data/processed/           generated (git-ignored): panel, codebook, name maps, audit
outputs/                  results, crosswalk, models/ (frozen), dryrun/ (DRYRUN only), release/
docs/                     report_template.md -> report.md, HUMAN_CHECKLIST.md, DEFENCE_NOTES.md,
                          NFHS6_PDF_CHECK.md, DATA_SOURCES.md, one-pager
config.yaml               all tunables and assumptions
DECISIONS.md              every judgement call, with reasons (D01-D40)
```

## Key outputs

| File | What |
|---|---|
| `outputs/SUMMARY.md` | what was built, counts, results, assumptions, data problems |
| `docs/report.md` | the report (generated; edit `docs/report_template.md`); `docs/report_values.json` gives each number's source |
| `outputs/results_nfhs4to5*.csv` | skill, MAE, RMSE with 95% CI per indicator and method (harmonised / stable / seen-state) |
| `outputs/paired_*_nfhs4to5*.csv`, `secondary_nfhs4to5*.csv` | ML vs RTM paired tests; secondary metrics |
| `outputs/crosswalk.csv`, `crosswalk_review.csv`, `harmonisation_report.md`, `harmonised_units.csv` | boundary lineage and analysis units |
| `outputs/models/` | frozen models and manifest (hashes) |
| `outputs/release/` | harmonised panel, units, codebook and crosswalk as CSV; data dictionary; sources and licence |
| `outputs/dryrun/` | **DRYRUN stand-in outputs, not NFHS-6 results** |
| `outputs/post_run/` | after Gate B: every NFHS-6 table and figure |

## Data sources

- **NFHS-4/5:** `India.csv` from https://github.com/SaiSiddhardhaKalla/NFHS, parsed
  from the IIPS NFHS-5 district fact sheets (with the NFHS-4 comparison column) and
  Census 2011 codes (about 65 newer districts lack codes). That repository states no licence.
- **NFHS-6:** Patra, N. K. (2026). *NFHS-6 Provisional District-Level Data
  (India): Independent Compilation.* GitHub. Parsed from the IIPS provisional
  State & District Fact Sheets. Attribution files: `data/raw/nfhs6/SOURCE/`.
- **Original publisher:** International Institute for Population Sciences (IIPS), Mumbai.

Details, licences and citations: `docs/DATA_SOURCES.md` and `outputs/release/SOURCES_AND_LICENCE.md`.
