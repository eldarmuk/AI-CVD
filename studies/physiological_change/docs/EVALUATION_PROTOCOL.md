# Evaluation protocol: preserving physiological change

Status: proposed protocol, 31 October 2026. The executable scaffold checks data and evaluation logic on synthetic examples. It does not implement or validate the proposed research method. Before a reference-data experiment, resolve the open specifications below and commit a dated protocol version. Record subsequent deviations and their reasons.

## Question and intended use

Can correction and filtering of wearable heart-rate estimates suppress real within-person changes, and can directly estimating uncertainty in the change preserve more reference-confirmed changes at a comparable error burden and coverage?

The intended research output is an interpretable change estimate with support information or an explicit inability to determine a change. This is a measurement-analysis task. A heart-rate change is not a diagnosis, arrhythmia label, emergency, or prediction of clinical deterioration. The initial evaluation is retrospective; clinical benefit and safe operational use would require separate evidence.

The main study uses paired wearable and independent ECG reference data. Any later telecare batch without an independent reference has a separate engineering evaluation. Its apparent physiological plausibility, agreement with a model, or operator escalation cannot substitute for measurement truth.

## Units, clocks and information boundaries

The independent sampling unit is the participant. A measurement session or a prespecified nonoverlapping pair of time intervals is the analysis unit. Keep participant identity consistent across sessions, body sites, devices and source-dataset derivatives. Multiple sensors on one person are not independent participants.

For each decision at time `t`, define a preceding baseline interval and a recent interval ending at or before `t`. The primary target is the difference between their prespecified ECG-derived heart-rate summaries. Choose the interval durations, separation and summary statistic during development after inspecting reference chronology, then freeze them. Do not assume that short waveform windows support six-hour trends or that the earlier alarm-study horizons apply here.

Record three distinct time concepts where available:

1. Measurement time and its time basis, including timezone, daylight-saving handling or source-relative clock.
2. Availability time: when a reading could have reached the decision system.
3. Decision time: the last instant at which model input is allowed.

A retrospective source that lacks availability timestamps supports a measurement-time analysis, not a verified online-latency claim. State this explicitly. Alignment offsets must be obtained from documented synchronization or a prespecified development procedure; never optimize an offset on held-out reference outcomes to improve agreement.

Use only current and preceding permitted wearable inputs. Future samples, centered smoothers, backward interpolation and reference ECG are prohibited as inference inputs in the primary scalar-only setting. The reference is available for training labels and evaluation, not as a hidden predictor. If a future study allows periodic reference calibration, define it as a distinct information setting with the same reference budget for every comparator.

## Reference eligibility and uncertainty

Audit the original reference-generation procedure before using provided labels: ECG sampling, synchronization, quality assessment, beat detection, artifact rejection, excluded rhythms and missing intervals. Prefer published or independently justified quality rules to rules optimized for model scores. Keep the index wearable's quality separate from ECG reference eligibility.

Require usable reference coverage in both target intervals. Estimate or characterize reference uncertainty where the dataset permits it, including disagreement between plausible reference-processing choices. Do not treat an optical estimate from another consumer wearable as equivalent to an ECG reference. Do not infer HRV from intermittent scalar heart rate.

Keep four categories distinct:

| Category | Evaluation handling |
|---|---|
| Reference unavailable or unusable | Exclude from accuracy denominators, but include in the reference-flow table and coverage accounting. |
| Reference supports increase, decrease or stability | Include in the corresponding main target denominator regardless of model abstention. |
| Reference cannot resolve the prespecified change threshold | Report as reference-indeterminate; do not turn it into a stable negative. Include continuous change analysis where its uncertainty is usable. |
| Reference is usable but model cannot decide | This is model abstention. Retain it in coverage and in the denominator of reference-confirmed changes. |

Prespecify a change magnitude threshold from the intended measurement question and reference repeatability. It is an engineering interpretation threshold unless independently justified as clinical. Freeze its value before external outcomes are opened. Use a small declared sensitivity grid rather than selecting the most favorable threshold afterward.

Do not choose target intervals because the model makes a dramatic prediction. Define the eligible decision schedule independently of the candidate predictions. For the main analysis, use nonoverlapping interval pairs within a participant. An overlapping-window sensitivity may be useful, but must not be presented as additional independent evidence.

## Development and untouched evaluation

Use a development cohort for source auditing, baseline reproduction, feature/method choices and calibration. Separate fitting and uncertainty calibration by participant. Preserve chronology within each participant; no window may cross a split boundary. Learned normalization, imputation parameters, error models, threshold selection and calibration all use development data only.

Before external evaluation, freeze and record:

- The source release, permitted inputs, cohort and reference rules, episode construction and target definitions.
- The code commit, dependency environment, seeds, preprocessing, fitted artifacts and checksums.
- The primary comparison, operating point and practically meaningful effect, with rationale.
- The full outcome table, uncertainty procedure, exclusions, subgroup analyses and sensitivity analyses.
- The identity and role of each dataset, including prior exposure and possible shared participants with derivative datasets.

Metadata and documentation may be inspected to establish external-dataset compatibility before freezing. Record any inspection of its values, labels or outcome summaries. Do not silently reuse an external set for model selection after an unsuccessful result. Any refit using it converts that version to development evidence and requires a different untouched evaluation for a confirmatory claim.

A later period from the same company is temporal evaluation within that service, not external-site validation. Report previously seen and new participants separately. A new file is not necessarily a new population, device, recording process or outcome definition. Repartitioning extensively explored historical telecare records does not create untouched evidence.

## Required comparator set

Start with simple causal methods and add complexity only for a demonstrated failure.

| Comparator | Purpose |
|---|---|
| Unchanged numerical estimate and its direct interval-to-interval change | Quantifies the consequence of any correction. |
| Last available reading, causal median and a robust causal smoothing/state model | Tests whether elementary handling of noise and gaps is sufficient. |
| Direct change regression with ordinary calibrated intervals | Strong baseline for choosing the correct target; direct change prediction alone is not a novelty claim. |
| Point-error model with a quality or uncertainty rejection rule | Tests the existing reading-accuracy objective. |
| Change-error model with a rejection rule | Tests the proposed change-focused objective. This is a candidate, not an established new method. |
| Applicable published scalar reliability or selective-correction method | Establishes the relation to closest prior work. Record inaccessible implementation details and any faithful reimplementation choices. |

Compare point-error and change-error rejection using the same underlying heart-rate estimator, inputs and development budget. First freeze the estimator to isolate the rejection objective. Then, if numerical correction is studied, evaluate correction on identical accepted episodes to separate value changes from selecting easier episodes.

Waveform- or movement-assisted methods belong to a separate information comparison. They show what richer acquisition might enable, but are not same-information competitors for a scalar-only method. State clearly whether public heart-rate inputs are proprietary device outputs or numerical estimates generated by a reproducible optical-signal algorithm. Neither should be assumed equivalent to the unknown telecare export mechanism.

## Outcomes and uncertainty

Report continuous change error, error in individual readings, and decision-level outcomes together. A reduction in individual-reading mean absolute error is insufficient evidence of preserved changes.

At minimum, report:

- **Change retention:** correctly reported direction among all reference-confirmed changes. Break down unreported changes into model abstention, an incorrect stability conclusion and the wrong direction.
- **False change burden:** change claims during reference-supported stability, per all eligible stable decisions and per stated observation exposure.
- **Wrong-direction burden:** increases reported as decreases and vice versa, with denominator stated.
- **Coverage:** accepted decisions, eligible person-time, participants and sessions with usable output; reference availability is reported separately.
- **Magnitude and calibration:** absolute change error, bias, interval coverage and width, plus participant-level distributions. Report the fraction of true changes attenuated beyond a prespecified tolerance.
- **Selection:** who loses coverage, which states are rejected, and whether apparent gains concentrate in one person, device, activity or reference-quality stratum.

The main operating point must be chosen using development data. On the external cohort, apply the frozen thresholds unchanged and report the achieved error and coverage. Matched-coverage curves on external data may be included as descriptive comparisons, clearly separated from this deployable fixed-threshold result. Do not tune a threshold to match an external error target and call that prospective performance.

Use paired participant-level resampling for method differences, retaining each participant's sessions and sensors together. A bootstrap of fixed predictions estimates conditional uncertainty; it does not include uncertainty from model fitting or prior method selection. If computationally feasible, add repeated participant-grouped fitting as a development sensitivity. Small participant counts can leave wide intervals even with many decisions; report them rather than infer precision from row counts.

Specify the main effect and a minimum useful improvement before the locked test. An illustrative operating-point diagram is not a power calculation. Determine attainable precision from development participants and reference-confirmed changes, and narrow the claim or stop if precision is inadequate. Keep exploratory subgroup findings distinct from prespecified primary inference.

## Natural error, induced acquisition stress and robustness

Primary evidence must include naturally occurring wearable errors while the independent reference remains usable. Run synthetic thinning, missing blocks, quantization or drift only as labeled secondary experiments. Such experiments test assumptions about a corruption process; they do not show that genuinely missing or incorrect telecare values have been recovered.

If observed telecare gap patterns motivate stress conditions, document how those patterns were obtained and why they are relevant. Use development-derived patterns and a declared protocol. Do not apply retrospective full-record extrema as though they were available at the decision time. Do not call a waveform-derived public scalar stream a validated simulator of a proprietary band.

Prespecified sensitivity candidates are reference-processing choices, plausible alignment uncertainty, interval duration, change threshold, gap length, participant influence, activity/motion and device. Limit the set to questions that could change the conclusion. Avoid a large search across cohorts, transformations and thresholds followed by reporting only the best result.

## Explainability deliverables

Each supported output should identify its baseline and recent intervals, observed values, data age, recording support, estimated change, uncertainty, decision and reason for abstention if any. These quantities must be computed from the exact inference inputs, with units and missingness visible. An explanation must not invent a disease mechanism or imply that a high confidence score proves accuracy.

For the paper, include synchronized wearable/reference examples covering a preserved genuine change, a false change, harmful correction, useful abstention and failure. Select examples by a documented rule and show representative failures as well as successes. Quantify correction and rejection harms over the full cohort so attractive examples cannot stand in for evaluation.

Prefer an inspectable baseline and decomposable decision to adding a post-hoc attribution tool without a concrete interpretation question. If a complex model is required, test the faithfulness and stability of its explanation; physiological plausibility alone does not validate it.

## Later company export

Follow [the specific company request](COMPANY_EVALUATION_REQUEST.md). A frozen model can be replayed locally on an unseen same-schema batch, with no change to live care. Without independent references, this supports only schema compatibility, input support, output coverage, distribution shift and execution-cost findings. Operator feedback can assess clarity and workflow burden. It cannot establish measurement accuracy, sensitivity to clinical deterioration, or safety.

If independent reference or outcome records happen to exist, define a separately versioned evaluation before looking at their results. Operational escalation labels and independently documented clinical outcomes answer different questions. Do not append a convenient alarm-prediction endpoint merely to obtain another favorable score.

## Gates and records

Proceed beyond feasibility only if the references and chronology support the target, the question survives comparison with prior work, and there is an actionable failure that strong simple methods do not already resolve. A useful contribution may be a generalizable negative finding, but it needs the same reference and evaluation discipline.

Stop or narrow if the apparent effect is driven by one participant, disappears against direct change regression, requires unrealistic synthetic corruption, or results primarily from losing difficult participants. Do not rescue the claim by changing the held-out cohort after inspecting results.

Maintain dataset cards, a split/exposure ledger, versioned configurations, immutable run manifests, a complete results table and a decision log including negative runs. Public tests use synthetic fixtures; patient-level outputs, clinical notes, pseudonym maps and restricted data stay outside version control. Aggregate release requires a disclosure review and appropriate reuse permission. Synthetic test success verifies software behavior, not research validity.

## Specifications still open before a real-data run

1. Final datasets, releases, reuse terms, modality and participant overlap audit.
2. ECG reference construction, alignment, uncertainty and eligibility rules.
3. Decision cadence, interval durations, minimum support and change threshold.
4. Development fitting/calibration splits and untouched external dataset registration.
5. Primary method contrast, fixed operating point, meaningful effect and attainable precision.
6. Exact baseline implementations, dependency locks and available compute.

Resolve these in a dated amendment. The protocol is intentionally not an instruction to run a full study with arbitrary defaults.
