# How Predictable Are District Health Outcomes? A Noise-Aware, Out-of-Time Test of Forecasting and Programme Surprise on India's NFHS-6

Team CodeFlow: Saurav S, Shikshith V, Thijl Thejaswi MN, Vishwas V Angadi.

{{run_status}}

Every result in this report is inserted from a result file by `scripts/build_report.py` and checked by
`scripts/check_report_numbers.py`; the source of each number is listed in `docs/report_values.json`.
Decisions are numbered as in `DECISIONS.md` (D01–D38). Analyses marked *exploratory* were added after the
pre-registrations D32 and D34.

## Abstract

District estimates from India's National Family Health Survey (NFHS) guide programme targeting, yet each
rests on a small sample and the next round arrives years later. We ask how far district outcomes can be
forecast one round ahead, and whether that depends on the indicator and the district rather than only on
the model. We trained forecasting models on the NFHS-4 (2015-16) → NFHS-5 (2019-21) transition, froze them
(training-data hash `{{train_hash}}`), and scored them once against NFHS-6 (2023-24). Boundary changes
were harmonised through a lineage crosswalk ({{n_train}} training districts, {{n_test}} test units).

In leave-one-state-out cross-validation on NFHS-4 → NFHS-5, regression to the mean (RTM) reached a pooled
skill of {{s45_rtm}} over persistence, and an elastic net with spatial-neighbour features reached
{{s45_en}}, with {{v45_en_rtm}} RTM in pooled error, but it was significantly better on only {{n45_en_better}} of {{n_ind}}
indicators after false-discovery-rate control. On the out-of-time NFHS-6 test the pre-registered primary
method ({{primary_method}}) reached a pooled skill of {{s56_primary}} (RTM {{s56_rtm}}), and the elastic net
had {{v56_en_rtm}} RTM. Conformal 90% intervals calibrated on NFHS-4 → NFHS-5 covered {{cov_split}}
(split) and {{cov_weighted}} (shift-weighted) of NFHS-6 values. {{n_surprises}} district-indicator pairs
were flagged as surprises at a 10% false-discovery rate.

## 1. Motivation

Indian health programmes are increasingly targeted at districts: the Aspirational Districts Programme
ranks districts on NFHS-type indicators, and state plans set district targets. A forecast of where a
district will be at the next survey is useful in two ways. It sets a benchmark against which a district's
measured change can be judged ("surprise"), and its accuracy tells planners how much of district
variation is predictable at all. Both uses fail if sampling noise is ignored: a district estimate based
on a few hundred respondents moves by several percentage points between rounds even when nothing
changes. This project therefore treats noise as part of the question. It asks how much skill is
possible given the noise (a noise ceiling), whether uncertainty intervals stay honest when the world
shifts between survey rounds, and which districts and indicators are forecastable.

The central claim tested is that **forecastability is a property of an indicator and a district, not
only of a model**.

## 2. Related work

*Small-area estimation.* District survey estimates are a classic small-area problem. The Fay–Herriot
area-level model (Fay & Herriot, 1979) and empirical-Bayes shrinkage (Efron & Morris, 1975) borrow
strength across areas; Rao & Molina (2015) review the field. We use the Fay–Herriot model as a denoiser
of the starting value.

*Regression to the mean.* Extreme district values tend to move toward the mean at the next measurement
partly because of sampling noise (Barnett, van der Pols & Dobson, 2005). A simple RTM regression is
therefore a strong baseline, and beating it is the relevant test for machine learning.

*Forecast evaluation.* We follow the out-of-sample, skill-score tradition (Hyndman & Athanasopoulos,
2021; Gneiting & Raftery, 2007) and compare methods with paired tests of loss differences in the spirit of
Diebold & Mariano (1995), using a state-clustered bootstrap. The ML methods are the elastic net (Zou &
Hastie, 2005) and gradient boosting with LightGBM (Ke et al., 2017).

*Uncertainty and surprise.* Conformal prediction gives distribution-free intervals (Vovk, Gammerman &
Shafer, 2005; Lei et al., 2018); weighted conformal prediction corrects for covariate shift (Tibshirani
et al., 2019). Surprise detection is a multiple-testing problem; we control the false-discovery rate
(Benjamini & Hochberg, 1995) and also compute conformal p-values (Bates et al., 2023).

*Pre-registration.* Fixing analyses before outcomes are seen guards against forking paths (Nosek et al.,
2018). Here the frozen models, the methods and the post-test analyses were committed to git before the
NFHS-6 evaluation, and the one-shot run is enforced in code.

## 3. Data

- **NFHS-4 and NFHS-5:** district values parsed from the IIPS NFHS-5 district fact sheets, which print
  the comparable NFHS-4 value beside each NFHS-5 value (community file `India.csv`).
- **NFHS-6:** the provisional IIPS State and District Fact Sheets (2023-24), in a community compilation
  (Patra, 2026). These values are provisional and were parsed from PDFs by a third party (section 11).
- **Cleaning and matching** (D05–D13): suppressed cells, the zero encoding of `India.csv`, name variants
  and indicator text are handled in code; indicators are matched across rounds by normalised text. The
  two copies of NFHS-5 (India.csv and the NFHS-6 file's own NFHS-5 column) were audited against each
  other; indicators and districts that disagree were excluded (D12).
- **Indicators:** {{n_ind}} percentage indicators in five domains (water and sanitation, women's
  empowerment and family planning, maternal care, child health, nutrition), chosen from NFHS-4/5
  coverage and comparability only (D14).

## 4. Boundary harmonisation

District boundaries changed between all three rounds. A lineage crosswalk records every parent → child
link with its source and a confidence level (D27–D31, `outputs/crosswalk.csv`). The used links form a
bipartite graph between NFHS-5 and NFHS-6 districts; each connected component is one analysis unit.

- **Training set (NFHS-4 → NFHS-5):** {{n_train}} districts whose NFHS-4 and NFHS-5 values refer to the
  same boundaries ({{n_train_added}} more than the base pipeline's stable set).
- **Test set (NFHS-5 → NFHS-6):** {{n_test}} units. Besides the stable districts it contains
  {{n_test_changed45}} districts whose boundary changed between NFHS-4 and NFHS-5 but not afterwards, and
  {{n_test_inherited}} districts that kept their name but lost territory in NFHS-6. The latter are scored
  against the forecast for the whole NFHS-5 district, carry extra variance and are flagged.
- **Stable-boundary subset:** {{n_stable}} districts with identical boundaries in all three rounds. Every
  result is reported on it as well.
- **Not resolved:** {{n_review}} cases are listed for review: {{n_rev_45}} districts that changed between
  NFHS-4 and NFHS-5 (no NFHS-4 value on their boundaries, so they cannot be used for training),
  {{n_rev_new}} new NFHS-6 districts whose parent is not recorded in the available data,
  {{n_rev_retained}} retained-name boundary changes, and {{n_rev_low}} low-confidence inferred parent.
  No lineage was invented: official Local Government Directory lineage could not be obtained
  automatically.

## 5. Noise model

Fact sheets do not publish sample sizes, so standard errors are **assumed**:
SE = 100·√(p(1−p)/n·deff), with per-district n by population group and design effect 2.0 (sensitivity
1.5–3.0; D16). Values inherited across a boundary change carry extra variance equal to 0.25 (remnant) or
0.5 (carved-out child) of the within-state between-district variance estimated on NFHS-5 (D29). These SEs
drive the noise ceiling, Fay–Herriot shrinkage, the normalised conformal scores and the surprise test,
so every noise-based statement inherits this assumption.

## 6. Methods

All methods predict round t+1 from round-t data only and were fitted on NFHS-4 → NFHS-5 only.

| Family | Methods |
|---|---|
| Baselines | persistence; national trend (mean change); state trend (state mean change); RTM (OLS of y_{t+1} on y_t) |
| Small-area | Fay–Herriot with state effects (REML); RTM on Fay–Herriot-denoised inputs (FH + RTM) |
| Machine learning | elastic net and LightGBM predicting the change, with own history, other indicators, spatial-neighbour features, state mean, region and state dummies |
| Ablations | own history only; + other indicators; + spatial features; full; full without state dummies |

**Validation.** Leave-one-state-out (LOSO) cross-validation is primary. A seen-state CV, in which held-out
districts' states appear in training, is reported as an optimistic reference because it exercises the
state dummies as the NFHS-6 test does, but shares each state's transition-specific shock.

**Metrics.** Skill = 1 − MSE(method)/MSE(persistence), pooled over indicators and as an unweighted mean
over indicators, with 95% state-clustered bootstrap intervals. Secondary metrics: skill net of the
national shift, the share of the variance of change that is sampling noise, and the noise ceiling
relative to persistence and to RTM.

## 7. Pre-registered analyses

Committed before any result they govern (verified by `scripts/preflight.py`):

- **D32 (models, tests, metrics).** The method list above; paired state-clustered bootstrap tests of
  ΔMSE against RTM for the elastic net, LightGBM and both no-state-dummy variants, with
  Benjamini–Hochberg adjustment across indicators; the secondary metrics; seen-state CV as a reference.
- **D34 (after the test).** Primary method = lowest pooled LOSO error on NFHS-4 → NFHS-5 (selected:
  {{primary_method}}). Conformal intervals at 90% in three variants (split, normalised by origin SE,
  covariate-shift weighted) calibrated on NFHS-4 → NFHS-5 residuals. Surprises: residual scaled by
  forecast-error plus sampling variance, BH at q = 0.1, compared with a naive top-k. Per-district noise
  ceiling and a district-versus-indicator decomposition of forecast error.
- **Exploratory, added after D32/D34 (D37).** Results by unit group (flagged rows separately), the
  comparison of NFHS-4 → NFHS-5 with NFHS-5 → NFHS-6 skill, and a structural-break check.

## 8. Results on NFHS-4 → NFHS-5 (cross-validation)

### 8.1 Skill

{{t45_main}}

National trend alone reaches {{s45_nat}}, so a large part of the skill of every trend-aware method is the
average change between rounds. RTM reaches {{s45_rtm}} (mean over indicators {{s45_rtm_mean}}); the
elastic net {{s45_en}} (mean {{s45_en_mean}}) and LightGBM {{s45_lgbm}}. Knowing each state's own change
(oracle, not a forecast) gives {{s45_oracle}}, no better than RTM. Fay–Herriot denoising alone adds
little ({{s45_fh}}; {{fh_deff15}} to {{fh_deff3}} across design effects 1.5–3.0) because it adds no trend.
On the stable subset the elastic net reaches {{s45_en_stable}} and RTM {{s45_rtm_stable}}.

{{fig45_skill}}

### 8.2 Is machine learning better than RTM?

{{t45_paired}}

Pooled over indicators, the elastic net has {{v45_en_rtm}} RTM (ΔMSE {{d45_en_rtm}}, 95% CI
[{{d45_en_rtm_lo}}, {{d45_en_rtm_hi}}]), but after Benjamini–Hochberg adjustment it is significantly
better on only {{n45_en_better}} of {{n_ind}} indicators ({{n45_en_better_stable}} on the stable subset).
LightGBM has {{v45_lgbm_rtm}} RTM. The gain is real but thin.

### 8.3 What do the features add?

{{t45_ablation}}

Other indicators add nothing beyond a district's own history; spatial-neighbour features carry the
gain over RTM, and state dummies add little in LOSO, where they cannot be exercised.

### 8.4 Noise and the national shift

{{t45_secondary}}

Sampling noise is {{noise_share_45}} of the variance of district change (pooled; mean over indicators
{{noise_share_45_mean}}). The noise ceiling is {{ceil_p_45}} relative to persistence but {{ceil_rtm_45}}
relative to RTM: a perfect forecaster of the true value could remove at most that share of RTM's error.

### 8.5 Seen-state reference

When states are seen in training, the elastic net reaches {{s45_seen_en}} (without state dummies
{{s45_seen_en_nostate}}), RTM {{s45_seen_rtm}} and state trend {{s45_seen_st}}. The jump comes from state
dummies learning the same transition's state shocks, which a real forecast cannot know; LOSO is the
honest estimate.

### 8.6 Development check of the post-test tools

On NFHS-4 → NFHS-5 alone (calibration on half of the states, evaluation on the other half), 90% split
conformal intervals for the elastic net covered {{dev_cov_split}} and shift-weighted intervals
{{dev_cov_weighted}}.

## 9. Out-of-time test on NFHS-6

The frozen models predicted NFHS-6 from NFHS-5 once. Test units: {{n56}} ({{n56_stable}} stable,
{{n56_changed}} changed between NFHS-4 and NFHS-5, {{n56_flagged}} flagged).

### 9.1 Skill

{{t56_main}}

The primary method ({{primary_method}}) reached a pooled skill of {{s56_primary}} (95% CI
[{{s56_primary_lo}}, {{s56_primary_hi}}]; stable subset {{s56_primary_stable}}). National trend reached
{{s56_nat}}, state trend {{s56_st}} and RTM {{s56_rtm}}.

{{fig56_skill}}

{{fig56_skill_stable}}

### 9.2 Results by unit group (exploratory)

Flagged units are scored against inherited forecasts and are shown separately.

{{t56_groups}}

### 9.3 Machine learning versus RTM

{{t56_paired}}

On NFHS-6 the elastic net had {{v56_en_rtm}} RTM (stable subset: {{v56_en_rtm_stable}} RTM), and
LightGBM had {{v56_lgbm_rtm}} RTM.

### 9.4 Noise and the national shift

{{t56_secondary}}

Sampling noise was {{noise_share_56}} of the variance of change, and the noise ceiling relative to RTM
was {{ceil_rtm_56}}.

### 9.5 Did cross-validation predict the test? (exploratory)

{{t56_compare}}

The rank correlation of methods' pooled skill between the two periods is {{cmp_spearman_methods}}; the
best method was {{cmp_best45}} in cross-validation and {{cmp_best56}} on NFHS-6. Across indicators, the
primary method's skill in the two periods correlates at {{cmp_spearman_ind}} (Spearman).

{{fig56_compare}}

### 9.6 Structural-break check (exploratory)

Comparing NFHS-5 → NFHS-6 with NFHS-4 → NFHS-5 per indicator, {{breaks_any}} indicators show a
significant change in at least one of: the mean change ({{breaks_shift}}), the RTM slope
({{breaks_slope}}) or the primary method's error scale ({{breaks_scale}}) (BH q = 0.05).

{{t56_breaks}}

{{fig56_break}}

## 10. Uncertainty, surprise and forecastability on NFHS-6

### 10.1 Conformal coverage

{{t56_coverage}}

For the primary method, nominal 90% intervals covered {{cov_split}} (split), {{cov_normalised}}
(normalised) and {{cov_weighted}} (shift-weighted) of NFHS-6 values.

{{fig56_coverage}}

### 10.2 Surprises

Of {{n_surprise_tests}} district-indicator pairs, {{n_surprises}} are surprises at a false-discovery rate
of 10%. A naive list of the {{naive_k}} largest residuals shares {{surprise_overlap}} of them; its picks
have a mean assumed SE of {{naive_se}} against {{bh_se}} for the FDR-controlled list, i.e. the naive list
favours noisy districts.

{{t56_surprises}}

{{fig56_surprises}}

### 10.3 Forecastability: indicator or district?

In an additive decomposition of log normalised squared error, indicators alone explain
R² = {{r2_ind}}, districts alone {{r2_dist}}, and both together {{r2_add}}.

Hardest indicators (lowest normalised skill):

{{t56_hard}}

{{fig56_map_skill}}

{{fig56_map_ceiling}}

## 11. Limitations

- **Assumed standard errors.** Sample sizes and design effects are assumptions (D16). The noise ceiling,
  Fay–Herriot shrinkage, normalised conformal scores and the surprise test all depend on them. The
  design-effect sensitivity is in `outputs/sensitivity_deff.csv`.
- **Provisional, community-parsed NFHS-6 data.** NFHS-6 values are provisional IIPS figures parsed from
  PDFs by a third party. An independent parser and a random spot-check against the official PDFs exist
  (`scripts/spot_check_nfhs6.py`, D33) but were **not run** because the PDFs were not in the repository;
  until they are, parsing errors in the NFHS-6 file cannot be excluded.
- **Partial harmonisation.** Only lineage supported by the available files was used. Districts that
  changed between NFHS-4 and NFHS-5 cannot be used for training, and they are not a random sample (D28).
  New NFHS-6 districts without a recorded parent are not scored. Inherited units are scored with an
  assumed extra variance (D29).
- **One training transition.** All models learned from a single transition, and much of the skill is
  the average national change; whether the 2015-16 → 2019-21 change pattern (with COVID-period fieldwork)
  carries over to 2019-21 → 2023-24 is exactly what section 9 tests.
- **Validation design.** LOSO does not exercise state effects, while the NFHS-6 test uses seen states;
  seen-state CV is optimistic.
- **Multiple comparisons beyond the pre-registration.** Exploratory analyses (D37) are not error-controlled
  across their own family and should be read as descriptive.

## 12. Future work

- **RQ5, Aspirational Districts:** comparing surprises in the programme's 112 districts with others needs
  the official district list and was out of scope.
- **Explanations and clustering:** SHAP values for the ML models and clustering of district trajectories.
- **Dashboard:** an interactive view of forecasts, intervals and surprises.
- **Full lineage:** official LGD lineage, district populations and NFHS-4 Census-2011 values would let the
  changed districts enter training and allow population-weighted aggregation (`data/lineage/README.md`).

## 13. Reproducibility

The training data and configuration are frozen in `outputs/models/` (training-data hash `{{train_hash}}`,
config hash `{{config_hash}}`, {{n_methods}} methods). Commands, in order: `scripts/run_pipeline.py`,
`scripts/preflight.py`, `scripts/gate_b.ps1` (runs the one-shot evaluation once), `scripts/post_run.py`.
The harmonised panel and codebook are released in `outputs/release/`. Every judgement call is in
`DECISIONS.md`.

## References

*Team: verify each reference against the original before submission.*

- Barnett, A. G., van der Pols, J. C., & Dobson, A. J. (2005). Regression to the mean: what it is and how to deal with it. *International Journal of Epidemiology*, 34(1), 215–220.
- Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023). Testing for outliers with conformal p-values. *Annals of Statistics*, 51(1), 149–178.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Efron, B., & Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association*, 70(350), 311–319.
- Fay, R. E., & Herriot, R. A. (1979). Estimates of income for small places: an application of James-Stein procedures to census data. *Journal of the American Statistical Association*, 74(366), 269–277.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- International Institute for Population Sciences (IIPS) & ICF (2021). *National Family Health Survey (NFHS-5), 2019-21: India.* Mumbai: IIPS. District fact sheets: http://rchiips.org/nfhs/
- International Institute for Population Sciences (IIPS). *National Family Health Survey-6 (NFHS-6), 2023-24: State and District Fact Sheets* (provisional). Mumbai: IIPS.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences*, 115(11), 2600–2606.
- Patra, N. K. (2026). *NFHS-6 Provisional District-Level Data (India): Independent Compilation* (v1.0.0). GitHub.
- Rao, J. N. K., & Molina, I. (2015). *Small Area Estimation* (2nd ed.). Wiley.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *Advances in Neural Information Processing Systems*, 32.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society B*, 67(2), 301–320.
