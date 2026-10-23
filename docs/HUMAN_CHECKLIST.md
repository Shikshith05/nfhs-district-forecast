# Human checklist before Gate B

Gate B runs the NFHS-6 evaluation **once**. After it runs, nothing about the models, the
training data, the test set or the pre-registered analyses may change without being logged
as a post-hoc change. Tick every box below first. Allow about two hours.

Counts are as of 2026-10-23 (training-data hash `d3f06bee428deeef`).

## 1. Boundary lineage cases: `outputs/crosswalk_review.csv` (165 rows)

Open the CSV and filter by `relation`. For each type, decide: *accept as is*, or *supply
official data* (section 4).

- [ ] **`boundary_changed_4to5_parent_unknown`: 132 rows, unresolved.** NFHS-5 districts
  with no NFHS-4 value on their boundaries (Telangana 31, Gujarat 17, Chhattisgarh 15,
  Assam 12, NCT of Delhi 11, Uttar Pradesh 10, Arunachal Pradesh 9, Meghalaya 8, Tripura 7,
  Punjab 4, Haryana 2, Madhya Pradesh 2, Maharashtra 2, West Bengal 2).
  - *Effect now:* they are not used for training. 121 of them are still forecast and scored
    on NFHS-6, because their boundary did not change again.
  - *Check:* skim the list for names that look like a plain rename rather than a boundary
    change. A rename would be a matching error. Report any to the team.
- [ ] **`new_district_parent_unknown`: 16 rows, unresolved.** New NFHS-6 districts whose
  parent is not recorded in our data:
  - Arunachal Pradesh: Kamle, Lepa Rada, Lower Siang, Pakke Kessang, Shi Yomi
  - Tamil Nadu: Chengalpattu, Kallakurichi, Mayiladuthurai, Ranipet, Tenkasi, Tirupathur
  - Telangana: Mulugu, Narayanpet
  - West Bengal: Alipurduar, Jhargram, Kalimpong

  *Effect now:* not scored. *Check:* accept, or supply the official parents (section 4).
- [ ] **`retained_name_boundary_changed`: 16 rows, medium, used.** NFHS-6 districts that kept
  their name but lost territory:
  - Arunachal Pradesh: East Kameng, East Siang, Lower Subansiri, Upper Subansiri, West Siang
  - Chhattisgarh: Bilaspur
  - Tamil Nadu: Kanchipuram, Nagapattinam, Tirunelveli, Vellore, Villupuram
  - Telangana: Jayashankar Bhupalapally, Mahabubnagar
  - West Bengal: Darjeeling, Jalpaiguri, Paschim Medinipur

  *Effect now:* each is scored against the forecast for the whole NFHS-5 district, with
  extra variance (D29), and flagged. The report also shows them as a separate group.
  *Check:* is it acceptable to score a reduced district against its old boundaries? If
  not, say so before Gate B; they can be dropped from the test set only before the freeze.
- [ ] **`split_candidate_single_flagged_parent`: 1 row, low, not used.**
  Bilaspur → Gaurela-Pendra-Marwahi (Chhattisgarh). This was inferred because Bilaspur is
  the only flagged district in the state. It is plausible, but no official source in the
  repo confirms it. *Check:* confirm it from the Chhattisgarh notification (2020) or LGD,
  or leave it unused. Pre-flight shows this case as its one crosswalk WARN.

## 2. Harmonisation summary: `outputs/harmonisation_report.md`

- [ ] Units: 571 training, 684 test, 547 stable. Do these match what you expect?
- [ ] The NFHS-5 audit excludes 10 districts from every set: Chandigarh, Lakshadweep, six
  Delhi districts, Maharajganj and Mahoba. Agree?
- [ ] Assumed extra variance for inherited values: 0.25 × the within-state variance for a
  remnant (D29). Acceptable?

## 3. Decisions D24–D40 (`DECISIONS.md`)

Read each one. Mark any you disagree with **before** Gate B.

- [ ] D24 package pins (Smart App Control) · D25 OS-independent hash · D26 run-once checks
- [ ] D27–D31 harmonisation: lineage evidence, units and sets, extra variance, weights, clustering
- [ ] **D32 pre-registration of methods, tests and metrics.** Primary comparison: each ML model
  vs RTM. Primary model rule: lowest pooled LOSO RMSE (selects the elastic net).
- [ ] D33 PDF spot-check design
- [ ] **D34 pre-registration of the post-test analyses.** Conformal 90%, BH q = 0.10, the
  surprise definition, forecastability.
- [ ] D35 pre-flight · D36 Gate B helper · D37 post-run (lists the analyses that are
  *exploratory*) · D38 report and number checker · D39 dry run · D40 release

## 4. Optional files (supplying any of them means re-freezing; see the warning)

| File | Format | What it recovers |
|---|---|---|
| `data/lineage/lineage_supplied.csv` | `data/lineage/README.md` §1 | parents of the 16 new districts and of Gaurela-Pendra-Marwahi; with populations, aggregated units instead of inheritance |
| `data/lineage/population.csv` | §2 | population weights, so split districts are aggregated rather than inherited |
| `data/lineage/nfhs4_census2011_values.csv` + `4to5` lineage | §3 | the 132 districts that changed between NFHS-4 and NFHS-5 enter **training** |
| `data/raw/pdfs/*.pdf` + `manifest.csv` | `docs/NFHS6_PDF_CHECK.md` | NFHS-6 CSV vs PDF agreement rate (does **not** change any model) |

> **Warning.** The three lineage files change the training set and/or the frozen test set.
> If you supply any of them, you must:
>
> 1. run the full `scripts/run_pipeline.py` (this re-freezes the models);
> 2. re-render the report (`scripts/build_report.py`);
> 3. log the change in DECISIONS.md;
> 4. commit everything;
> 5. re-run the pre-flight.
>
> The PDFs only add an audit, so no re-freeze is needed:
>
> 1. run `scripts/spot_check_nfhs6.py`;
> 2. commit `outputs/nfhs6_spot_check.*`;
> 3. log the agreement rate in DECISIONS.md.
>
> If agreement is poor, discuss it before Gate B.

- [ ] Decided which optional files to supply (or none), and done the steps above.

## 5. Final checks, then Gate B

- [ ] Read `docs/report.md` through section 8 (the NFHS-4 → NFHS-5 results). It is the
  report that will be completed automatically.
- [ ] Look at `outputs/dryrun/DRYRUN_report.md`. It shows what the final report will look
  like, but with **meaningless stand-in numbers**.
- [ ] `git status` is clean, and you are on the commit you want to freeze (normally `main`).
- [ ] `.venv\Scripts\python scripts\preflight.py` shows `PRE-FLIGHT PASSED` (2 WARNs are
  expected: the Bilaspur lineage, and D34 sharing a commit with the development outputs).
- [ ] Then, once: `powershell -ExecutionPolicy Bypass -File scripts\gate_b.ps1` → type `YES`.
- [ ] After it finishes: `git push origin main freeze-v1`, then
  `.venv\Scripts\python scripts\post_run.py`, then commit `outputs/post_run/`,
  `docs/report.md`, `docs/report_values.json` and `outputs/release/`.

If Gate B stops with an error, **read the message and do not re-run it blindly.** Before
scoring, it tells you how to remove the tag. After scoring, the run counts and must be
logged.
