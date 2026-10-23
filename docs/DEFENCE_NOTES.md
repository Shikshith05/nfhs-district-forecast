# Defence notes: likely questions and plain answers

Short answers for the viva or for mentor review. Numbers refer to NFHS-4 → NFHS-5
cross-validation unless stated; NFHS-6 numbers appear in `docs/report.md` after Gate B.
Decision numbers (Dxx) point to `DECISIONS.md`.

## The design

**Why not just train on all three rounds?**
Because then we could not test honestly. We trained only on NFHS-4 → NFHS-5, froze the
models (git tag `freeze-v1`), and scored NFHS-6 once. The code refuses to run the test a
second time, refuses to run on uncommitted changes, and refuses if the training data
changed (D26). A test you can re-run until it looks good is not a test.

**How do we know NFHS-6 did not leak into training?**
Every training function checks that no round-6 row reaches it and stops if one does. Tests
prove this, including one that scrambles NFHS-6 values and shows the predictions do not
change (D20, D35). NFHS-6 was used only for metadata before the test: which districts
exist and the boundary flags. It was never used for values.

**What was decided before seeing NFHS-6?**
- All methods, the paired tests and the secondary metrics (D32).
- Every post-test analysis: intervals, surprises, forecastability (D34).

Both were committed to git before any result they govern. `scripts/preflight.py` checks
the commit order. Anything added later is labelled *exploratory* in the report: results
by unit group, the comparison with the 4→5 results, and the structural-break check (D37).

**Why is "skill" measured against persistence?**
Persistence ("next value = last value") is the forecast you get for free. Skill = 1 −
MSE(method)/MSE(persistence): 0 means no better than persistence, 1 means perfect.

## The results

**Did machine learning win?**
Barely.
- **Pooled:** RTM reached about 0.61. The elastic net with spatial-neighbour features
  reached about 0.64. That pooled gain is statistically significant.
- **Per indicator:** after correcting for testing 23 indicators, the elastic net is
  significantly better on only 2 (4 on the stable subset).
- **LightGBM:** not significantly better than RTM.

We report this as a finding: most of what is predictable is the national average change
plus regression toward the mean.

**What is regression to the mean, and why is it so strong here?**
A district measured unusually high in one round tends to come out lower in the next, and
vice versa. Part of the reason is noise, because a lucky sample does not repeat. A simple
line, next = a + b × last with b below 1, captures this. With noisy small-area estimates
it is hard to beat.

**What did the features add?**
- A district's own history alone matches RTM.
- Adding the other 22 indicators adds nothing.
- The neighbouring districts' values (spatial features) provide the gain.
- State dummies add little in cross-validation, which cannot test them.

**Why are there two "state" numbers, LOSO and seen-state?**
Leave-one-state-out CV hides a whole state, so it never tests what the model learned
about that state; the real NFHS-6 test includes every state. Seen-state CV shows
0.76 for the elastic net, but it is optimistic: it shares each state's 2015→2020 shock
between training and testing, which a real forecast cannot know (D32). We trust LOSO
and the NFHS-6 test.

## Limitations, and what each one means

**"The standard errors are assumed."**
- *What it means:* the fact sheets don't publish sample sizes, so we assumed them (about
  1,000 women or 250 recent mothers per district) and a design effect of 2.
- *What it affects:* everything about noise: the noise ceiling, Fay–Herriot shrinkage,
  interval widths in the normalised variant, and which districts count as surprises.
- *What it doesn't affect:* the forecasts of RTM and the ML models, and their
  skill scores.
- *How we checked:* a sensitivity analysis over design effects from 1.5 to 3 (D16).

**"The NFHS-6 data are provisional and community-parsed."**
- *What it means:* IIPS may revise the numbers, and a third party extracted them from PDFs.
  Errors in that extraction would look like forecast errors or surprises.
- *What we did:* audited the NFHS-5 values printed in the same file against our other
  source (95% exact agreement), and built an independent PDF parser plus a random
  spot-check (D33).
- *What's still missing:* the spot-check has not been run because the official PDFs
  are not in the repository. Until it runs, say so.

**"Harmonisation is partial."**
- *What it means:* we only used boundary links we could prove from our files.
- **132 districts** changed between 2015 and 2019. They cannot be used for training,
  because they have no NFHS-4 value on their current boundaries. They are not random:
  they cluster in states that created new districts (Telangana, Gujarat, Chhattisgarh,
  Assam, Delhi).
- **16 new NFHS-6 districts** are not scored, because we could not cite their parents.
- **16 districts** lost territory. They are scored against their old boundaries, with
  extra allowed noise, and shown separately.
- *What would fix it:* official LGD lineage files and populations (`data/lineage/`).

**"Only one training transition."**
Everything was learned from 2015-16 → 2019-21, a period that included COVID-era
fieldwork. Whether that pattern of change carries over to 2019-21 → 2023-24 is
exactly what the test measures; a large drop in skill would be a finding about
non-stationarity, not a bug.

**"Many analyses, many tests."**
- *Controlled:* the pre-registered tests use Benjamini–Hochberg correction.
- *Not controlled:* the exploratory analyses have no family-wide correction. Read them
  as descriptive.

## Methods questions

**Why Fay–Herriot, and why does it score so low?**
Fay–Herriot shrinks each noisy district value toward its state average by an amount
that depends on how noisy it is. Used alone it predicts "no change", so its skill
is low (about 0.08). It helps as an input: RTM on Fay–Herriot-denoised values (FH + RTM)
beats plain RTM slightly.

**What is a conformal interval?**
- *Calibration:* we take the errors our method made on 2015→2020 and use their 90th
  percentile as the interval half-width.
- *Guarantee:* if the future behaves like the past, 90% of new values fall inside.
- *What we test:* whether that holds for 2023-24.
- *The weighted variant:* re-weights past errors to look like the districts being
  forecast, which helps if the mix of districts has shifted.

**How is a "surprise" different from "the biggest miss"?**
A big miss in a district with a tiny sample is expected. We divide each miss by how big
a miss should be, given both the usual forecast error and that district's sampling
noise. We then keep only cases that survive a false-discovery-rate limit of 10%. The
naive "biggest misses" list mostly picks noisy districts, and the report shows this.

**What is the noise ceiling?**
It is the best skill any method could reach if it predicted the true district value
perfectly, because the measured value still contains sampling noise. Relative to RTM it
is about 0.83. So the gap between our methods and what is possible is large in
principle, but the assumed noise sets the size of that gap.

**Why the stable-boundary subset?**
The 547 districts with identical boundaries in all three rounds need no harmonisation
assumptions. Every result is reported on it too, so a reader can see that harmonisation
did not drive the conclusions.

## Process questions

**How do I know the report numbers are right?**
Every result number in `docs/report.md` is inserted from a result file. A checker
re-derives each one from its source file and fails if any differs. It also fails if a
hand-typed decimal appears in the prose (D38). `docs/report_values.json` lists the file
behind every number.

**Could the stand-in test results be confused with the real ones?**
No. Every stand-in file:
- is named `DRYRUN_…` and lives in `outputs/dryrun/`;
- carries a `run_mode = DRYRUN` column;
- has watermarked figures.

The real pipeline refuses such files, and tests fail if the real report ever refers to
them (D39).

**What is out of scope?**
- *RQ5:* Aspirational Districts, which needs the official district list.
- *Model explanations:* SHAP values and clustering.
- *A dashboard.*
- *Full official lineage.*

These are listed as future work, not claimed.
