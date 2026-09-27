# Problem 8 decision after the predefined controls

Final review 2026-09-27: **Outcome A — close Problem 8 with R0 as the canonical
reference architecture.** The completed head-convergence diagnostic passed
independent verification. Correcting the value-access control changes its
scientific interpretation, but does not justify more architectural complexity
or another endpoint-adaptation experiment before freezing the reference.
No encoder was retrained and the frozen study remains unchanged.

The final decision and corrected tables below supersede the historical pending
decision later in this document. Retain the earlier results as the original
bounded study, and report the numerical sensitivity analysis separately.

## Final verification of the completed numerical diagnostic

`head-convergence-v1/complete.json` reports completion and all stationarity gates
passing. All **14 diagnostic artifact hashes** match; all **117 original study
and control-analysis hashes** also still match. The original R0/R1 SSL and risk
checkpoints, sampling/masking plans, source/config fingerprints, six completion
markers and original results remain unchanged.

An independent, read-only audit (`verify_head_convergence.py`) verifies all 12
diagnostic heads. Fit and assessment sample IDs match the original ordered plans
and saved embedding IDs; fit inverse-inclusion weights match exactly. Patient
membership and folds are rechecked through the original training-only scope
audit. Labels are reconstructed from the same immutable episode-linked sample
plans used by the diagnostic source; assessment labels also match the original
saved controls. Labels are not separately duplicated in the diagnostic files.
The hashed diagnostic source uses those plans directly, and its only optimizer
operates on a newly created logistic layer over saved embeddings. No patient
shards, canonical validation/test outcomes, or encoders enter this refitting.

All 12 heads reproduce their saved scores exactly in float64. Weighted objectives
and gradients were recomputed independently from the saved coefficients in the
standardized training coordinates, including the unchanged L2 term. Maximum
absolute gradients range from **8.27e-11 to 4.76e-10**, below the predeclared 1e-8
stationarity gate. Fits took 39–105 iterations, within the diagnostic 500 cap;
all objectives decreased. The recorded fitting/report interval was **10.30 seconds**,
excluding the initial original-artifact audit. Results are saved privately in
`head_convergence_verification.json`.

No private optimization was run during this review. Eight focused tests pass,
including independent objective/gradient reproduction and detection of a changed
bias, plus the existing synthetic control-analysis and alignment tests.

## Corrected development results

Each cell is **AUROC / AP x 100,000**. Values are inverse-inclusion-weighted
training-fold assessment estimates; the weighted prevalence remains 3.1022 per
100,000. They are not canonical test-set or full-stream clinical utility results.
Use every converged head consistently, not the better of original and corrected.

| Representation/head | Fold 0 | Fold 1 | Fold 2 | Pooled |
|---|---:|---:|---:|---:|
| SSL R0 joint, converged | .6160 / 5.5061 | .6227 / 4.1814 | .6632 / 5.4871 | .6310 / 4.6792 |
| SSL R1 joint, converged | .6014 / 4.4838 | .5916 / 3.8228 | .6628 / 5.9199 | .6147 / 4.4850 |
| Process-only, converged | .5903 / 4.1590 | .5910 / 4.1532 | .6538 / 5.8072 | .6058 / 4.3075 |
| SSL R0 value-access, converged | .6108 / 5.9786 | .6124 / 3.8511 | .6747 / 5.2942 | .6251 / 4.4280 |
| Frozen-random R0, original head | .6158 / 4.4563 | .6109 / 4.2547 | .6551 / 5.6821 | .6238 / 4.6001 |

The random control was not refitted: its embeddings were not saved. Its hash,
patient/sample alignment and recorded metrics remain verified, but independent
score replay and strict-gradient convergence are unverified. Therefore the
corrected-SSL versus original-random comparison is a qualified sensitivity
comparison, not an exactly solver-matched refit. The original matched-budget
comparison also showed no established SSL benefit.

| Pooled result | Original AUROC | Converged AUROC | Original AP x 100,000 | Converged AP x 100,000 |
|---|---:|---:|---:|---:|
| SSL R0 joint | .63068 | .63099 | 4.67502 | 4.67923 |
| SSL R1 joint | .61487 | .61475 | 4.49135 | 4.48497 |
| Process-only | .60560 | .60577 | 4.30352 | 4.30748 |
| Value-access | .58723 | .62507 | 3.92725 | 4.42803 |

The optimizer defect **materially affected value-access**: pooled AUROC increases
by .03784 and AP by .50078 per 100,000. Fold 2 changes from .5000 / 3.1139 to
.6747 / 5.2942. The other pooled results barely change. This overturns an
interpretation that process-only generally outperforms value-access, and removes
the earlier apparent clear pooled joint advantage over value-access.

## Strength of the corrected evidence

**Consistent directional evidence:** value-access AUROC exceeds process-only by
.02051, .02142 and .02096 across the three folds. Joint R0 also exceeds process-only
AUROC in all folds (+.02572, +.03180, +.00945). Thus access to the value-containing
temporal representation appears to contribute useful incremental ranking signal
over fixed process summaries in this development study. It is no longer defensible
to dismiss the value representation based on the failed fold-2 fit.

**Weak or mixed evidence:** value-access AP exceeds process-only only in fold 0;
joint R0 AP exceeds process-only in folds 0 and 1, but is lower in fold 2. Joint
R0 versus value-access is also mixed: AUROC favors joint in two folds and
value-access in fold 2; AP favors joint in folds 1 and 2 and value-access in fold 0.
The small pooled joint advantage does not establish that concatenating process
summaries consistently improves over value-access for every ranking metric.

**Unresolved interpretation:** value-access is not purely physiological: its
encoder necessarily contains masks, recency and temporal observation structure.
It also includes Steps and evidence flags. These controls distinguish learned
value-access representations from fixed process summaries, not a causal effect
of physiological magnitudes alone. Process-only remains a credible predictor
and wins AP in two folds against value-access. Retain observation-process
dependence as a central limitation and process-only/missingness-stratified
evaluation as planned ablations. Clinical usefulness, transportability to another
monitoring workflow and gains specifically attributable to physiology remain
unresolved.

For context, descriptive 500-repetition paired patient-cluster 95% intervals
(same seed 20260926) are below. They condition on fixed fits/sampled windows,
do not account for retraining uncertainty, and are not multiplicity-adjusted.
Statistical significance is not a gate for this architecture decision.

| Corrected contrast | Pooled AUROC difference (95% interval) | AP difference x 100,000 (95% interval) |
|---|---:|---:|
| Joint R0 minus process-only | +.02522 (-.00667, +.05211) | +.37175 (-.33481, +1.61298) |
| Value-access minus process-only | +.01930 (-.02097, +.06042) | +.12055 (-.77775, +1.50776) |
| Joint R0 minus value-access | +.00592 (-.01912, +.02684) | +.25120 (-.37456, +.79283) |
| Joint R0 minus original random | +.00716 (-.01733, +.02794) | +.07914 (-.51417, +.88908) |
| R1 minus R0 | -.01624 (-.04060, +.00678) | -.19426 (-1.09359, +.28465) |

**SSL conclusion:** pretraining learns reconstruction, but meaningful downstream
benefit over frozen-random features is still not demonstrated. Corrected SSL R0
has slightly higher AUROC in all folds; AP improves only in fold 0. The numerical
repair does not change that conclusion. This is not evidence that SSL is useless,
and the random-head limitation prevents a definitive equivalence claim.

## Final Problem 8 decision: A

**Close Problem 8 with R0 as the canonical reference architecture.** Preserve the
approved primitive projection, missingness-aware single encoder, deterministic
empty-window handling, fixed process summaries and frozen logistic-head design.
Treat reliable head convergence as an implementation requirement in future
protocols; do not retroactively change the frozen original study or its checkpoints.

R1 has no consistent benefit, and improved reconstruction still does not predict
better endpoint ranking. The corrected controls show useful value-access ranking
signal rather than a concrete failure requiring representation-to-endpoint
adaptation. With only 124 training episodes, final-block adaptation would introduce
additional fitting flexibility without evidence that it is needed to settle this
bounded architecture choice. Do not add that experiment merely because the
pooled AUROC is modest or intervals include zero. No R2 cross-attention or R3 VAE.

There is **no remaining architecture-selection experiment**, so no training
command, runtime, resume mode or completion marker is applicable. Do not rerun
`head_convergence` against its completed directory. Problem 9 has not begun.
Closing this development choice is not a claim of clinical validation, SSL
superiority, or a predominantly physiological predictor.

## Historical review before the convergence diagnostic

The following records the prior pending decision and original results. Its
requests to run the numerical diagnostic and keep Problem 8 open are superseded
by the verified final decision above.

## Artifact verification

Run: `runs/model_studies/r0-r1-bounded-3fold-v1`.
The control analysis completed in approximately 15.4 minutes (artifact timestamp
interval). `analysis/comparison.json` and `analysis/artifacts_sha256.json` exist.
All **117 listed artifact hashes** match, including nine control heads, predictions,
original checkpoints, plans, metadata and six model completion markers.
The root completion marker and all original fit checks also pass.

The short read-only audit saved `control_verification.json` outside the immutable
`analysis/` directory. It independently checks the current frozen config, task and
canonical run/TRAIN-export fingerprints; original implementation hashes and
approved resume receipt; reconstructed training-only grouped folds; authorized
SSL/scaler/checkpoint/head/assessment populations; sample order, patient identity,
labels, inverse-inclusion weights, evidence flags and shared corruption hashes.
No canonical validation/test outcomes or patient shards were read.

All recorded fold/stratum metrics reproduce from saved predictions. Original
R0/R1 heads reproduce their scores. All six process/value-access control heads
also reproduce their float32 scores exactly using the original feature slicing
layout. Their weighted training embedding means reproduce independently.
The control fitting source uses exactly the saved head samples, assessment samples,
weights, L2 and iteration limit from the original study; head scope identities agree.

**Audit limitations:** random-control embeddings were not persisted. Its head,
checkpoint and predictions pass hashes, scope, alignment and metric checks, but
independent replay of its scores was not performed. Doing that requires another
training-partition embedding pass. Control head files do not separately record
fit sample IDs/weights; these are bound through the hashed analysis source and
original saved plans, rather than a separate fit receipt. Do not describe this as
complete independent replay of every control. No evidence of patient leakage or
configuration drift was found within the verified scope.

Configuration fingerprint:
`e824e78604684e9d5adfa6ddb5ad504bb6ae8f6d5a93c7ece49f8911ebcb1850`.
Analysis manifest fingerprint:
`3a1460ab02dd67b4f308fae6715e7ad706845d849aa5e15095085eddb7b0653c`.

## Results

Each cell below is **AUROC / AP x 100,000**. These are inverse-inclusion-weighted
development estimates from the three canonical-training patient holdouts, not
natural-test or full-stream performance. The pooled weighted prevalence is
3.1022 per 100,000 windows. There are 9,218 sampled assessment windows, 514 positive
windows, 8,704 held patients, 107 event patients and 124 linked episodes. Repeated
positive windows are not independent clinical events.

| Representation/head | Fold 0 | Fold 1 | Fold 2 | Pooled |
|---|---:|---:|---:|---:|
| SSL R0 joint | .6161 / 5.4709 | .6226 / 4.1823 | .6623 / 5.4952 | .6307 / 4.6750 |
| SSL R1 joint | .6006 / 4.4745 | .5913 / 3.8095 | .6634 / 6.0029 | .6149 / 4.4913 |
| Frozen-random R0 | .6158 / 4.4563 | .6109 / 4.2547 | .6551 / 5.6821 | .6238 / 4.6001 |
| Process-only | .5895 / 4.1511 | .5913 / 4.1503 | .6539 / 5.8007 | .6056 / 4.3035 |
| SSL R0 value-access | .6102 / 5.6949 | .6124 / 3.8553 | **.5000 / 3.1139** | .5872 / 3.9273 |

Additional descriptive paired patient-cluster bootstrap intervals use the same
500 repetitions and seed 20260926 as the original comparison. They condition on
fixed fitted models and sampled windows, omit retraining/seed uncertainty, and
are not adjusted for multiple comparisons. No thresholds or selection criteria
were changed. Pooling scores from different fold heads also depends on comparable
score scales; per-fold results take priority.

| Joint SSL R0 minus control | Pooled AUROC difference (95% interval) | AP difference x 100,000 (95% interval) |
|---|---:|---:|
| Frozen-random R0 | +.00685 (-.01786, +.02780) | +.07493 (-.50229, +.86807) |
| Process-only | +.02508 (-.00664, +.05241) | +.37150 (-.33763, +1.59721) |
| Value-access | +.04345 (+.01061, +.07828) | +.74777 (+.01676, +1.45962) |

The last row is **not evidence against value information**: it includes a control
that did not take an optimizer step in fold 2. The original R1-minus-R0 pooled
AUROC difference remains -.01581 (95% interval -.04056 to +.00758).

**SSL usefulness:** reconstruction was learned, but reproducible downstream benefit
is not established. R0 SSL AUROC is slightly higher than random in all folds
(+.00028, +.01172, +.00717); AP improves in only one of three folds. The pooled
intervals include zero. Frozen-random is a supervised logistic head on an untrained
encoder, not an end-to-end supervised-from-scratch neural model. Its joint vector
also includes the same fixed process summaries. More SSL complexity is unsupported.

**Observation behavior versus values:** process-only retains much of joint ranking.
Joint R0 gains modest AUROC in all folds (+.02653, +.03130, +.00844), but AP improves
in two folds and declines in fold 2; pooled uncertainty includes zero. This is a
substantial observation-process signal, not proof that all prediction is driven by
it. Incremental physiological information is not established by these results.
Value-access still includes minimum masks/recency and evidence flags: it is not
pure physiology. A joint-versus-value comparison tests adding explicit process
summaries, not removing all observation information.

**R1:** no reproducible improvement over R0. Its slightly better reconstruction in
all folds does not translate into better Level-3 ranking. Reconstruction models
common observed measurements; the rare retrospective alarm endpoint need not
align with that objective. Neither cross-attention R2 nor variational R3 is earned.

## Empty windows and clinical usefulness

AUROC in windows with no observed physiological channels:

| Model | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|
| Joint SSL R0 | .6180 | .5415 | .6989 |
| Frozen-random R0 | .6158 | .5448 | .7179 |
| Process-only | .6585 | .5632 | .6549 |
| Value-access | .5134 | .4870 | .5000 |

These strata contain only 17, 21 and 10 positive windows. Steps may still be
observed; no-physiology and no-primitive are distinct. For wholly unobserved
primitives the R0 value pool is exactly zero and context remains. This supports
the need to retain process baselines and missingness strata, not a claim of
physiological deterioration detection.

Within physiology-observed windows, joint R0 AUROC is .5708/.5997/.6227 versus
process-only .5355/.5570/.6033. The consistent direction is suggestive, but it does
not isolate physiology causally or establish clinically meaningful improvement.
No model emits a probability >= .5: recall is zero and precision undefined at
that diagnostic threshold. Weighted pooled log loss is .00035163 for SSL R0,
.00035129 random, .00035138 process-only, .00035246 value-access. These tiny
differences, and Brier scores near prevalence, do not establish calibration or
utility. No predeclared clinical alert policy was evaluated; episode sensitivity
and false alerts per supported patient-day remain unavailable.

## Numerical defect and the one required next check

Fold-2 value-access has exactly zero fitted slopes and unchanged objective
(.0003524528357369067). Every score equals the fitted intercept probability.
An independent algebraic check on its saved training embeddings/labels/weights
gives initial maximum absolute gradient **9.24698e-6** and initial directional
derivative **-8.44778e-10**. The installed PyTorch LBFGS default is
`tolerance_change=1e-9`; it exits when the directional derivative is greater than
`-tolerance_change`, before line search. Thus the gradient is nonzero but the
rare-outcome objective scale triggers premature stopping. This is a numerical
failure of the control fit, not representation collapse. Other head convergence
was not proven by reaching a fixed iteration cap either.

**Only recommended next experiment:** a numerical convergence sensitivity audit
of the same saved R0, R1, process-only and value-access embeddings. Keep the exact
weighted logistic objective, L2, folds, samples and weights. Use fixed stricter
stopping tolerances (`1e-10` gradient, `1e-15` change), a 500-iteration maximum,
and report final gradients. Require maximum absolute gradient <=1e-8 for the
stationarity gate. This gate checks numerical convergence, not model performance.
Apply it to all four representations in all folds; never choose the better of
old/new scores. Do not change or overwrite the frozen study. Random embeddings
are unavailable, so this audit cannot revise the SSL-versus-random conclusion.
It does not train encoders and is not Problem 9.

This check precedes endpoint adaptation because the current control failure
prevents a fair interpretation. If it preserves the findings, close architecture
selection with R0 as a modest reference and report no established SSL advantage;
do not require a positive result before freezing the architecture.

## Requested alternatives, considered but not selected

Compute ranges are planning estimates on four CPU threads, not measured runs;
the prior full bounded study took about 98 active minutes. All alternatives would
need a separate prospectively fixed protocol and training-only patient folds.

| Option | Hypothesis and ranking plausibility | Estimated three-fold compute | Overfitting/design assessment |
|---|---|---|---|
| A: limited final-block adaptation | Align the representation with Level-3 rather than reconstruction; plausible if the frozen representation misses task direction | 30–120 min for a small fixed schedule; needs cached-window design | High with ~83 fit episodes/fold. R0 has one recurrent block: unfreezing it is not a trivial top-layer change. Must define exactly which parameters and inner-training stopping before running. |
| B: small nonlinear head | Learn interactions hidden from a linear head; plausible but unsupported by current controls | 5–20 min using cached embeddings | Moderate/high with 124 total episodes; strongly regularize and fix capacity. Easier isolation than A, but first establish linear convergence. |
| C: longer SSL | Test undertraining; limited rationale because selected held losses mostly bottomed near draw 1800 and downstream gains are absent | 30–60 extra min for one additional 2000-draw R0 budget/fold, excluding setup | Low direct label overfit, high risk of improving only reconstruction. Not currently warranted. |
| D: short trajectory masks | Demand temporal inference rather than isolated-cell reconstruction; uncertain under sparse observations | 45–90 min for a new bounded R0 pretraining pass | Changes corruption exposure and needs target-count matching. Sparse blocks may provide no targets. No current evidence it fixes endpoint alignment. |
| E: causal patient-relative deviations | Capture departures from an individual's prior observed baseline; plausible | Tens of minutes to hours of feature/projection work plus fitting | Must use only prior history, define cold starts and sufficient baseline support, and version the projection. Risks weak baselines and monitoring-pattern confounding; cannot alter the frozen canonical run. |

No additional architecture experiment is authorized or selected by this memo.

## Manual command for the numerical check

The command is implemented and synthetic-tested, but **has not been run privately**.
Expected time: approximately 1–5 minutes, conservatively allow 10 minutes. It
loads small saved embeddings and audits saved artifacts, not patient shards.
It refits only new diagnostic logistic heads; it never retrains R0/R1 encoders.

```powershell
Set-Location 'C:\Users\eldar\Projects\AI-CVD'
$numericalLog = 'runs/model_studies/r0-r1-bounded-3fold-v1/head-convergence-v1.log'
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -u -m src.architecture_study.head_convergence --run runs/model_studies/r0-r1-bounded-3fold-v1 --output runs/model_studies/r0-r1-bounded-3fold-v1/head-convergence-v1 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -LiteralPath $numericalLog -Value $line -Encoding utf8
}
if ($LASTEXITCODE -ne 0) { throw 'Numerical audit failed; preserve all outputs.' }
```

Completion: `head-convergence-v1/complete.json` with `status: complete`, all listed
hashes intact, and inspect `all_stationarity_pass`. A completed run with a false
stationarity gate is not a successful convergence result. Preserve the entire
subdirectory and the log. **Not resumable:** it refuses an existing output
directory and preserves partial work; do not delete or overwrite it. Request
recovery if interrupted. Original study/analysis remain untouched.

```bash
tail -F /c/Users/eldar/Projects/AI-CVD/runs/model_studies/r0-r1-bounded-3fold-v1/head-convergence-v1.log
```

## Validation and current decision

- Read-only private artifact verification passed; all 60 fold/stratum metric rows
  reproduce, six control heads replay exactly, pooled controls and paired intervals
  recomputed. Random-head replay limitation remains explicit.
- Eight focused analysis tests pass, including fictional end-to-end controls,
  identity/weight/label tampering, metrics, evidence strata, and rare-outcome
  numerical convergence. No private fitting or test-set evaluation was executed.
- Frozen task/config, canonical dataset and original model artifacts are unchanged.
- **SSL:** learns reconstruction; downstream benefit not established.
- **Physiology beyond observation:** suggestive direction in some comparisons,
  not established; value-access fold 2 is numerically invalid as a fitted control.
- **Preferred architecture:** R0 provisionally; no R2/R3 progression.
- **Problem 8:** open only for the stated numerical head-convergence check before
  final architecture closure. Do not start Problem 9.
