# Research plan: preserving physiological changes in wearable monitoring

**Status:** protocol development and executable synthetic foundation.
**Created:** 31 October 2026. **Planning horizon:** November 2026–January 2027.
**Branch:** `research/hr-change-reliability`.
**Public base:** `a81239ac4419f0ea0c5d36a8850e0063c6e2bc91`.

This file is the authoritative starting point for future work. It describes a
new measurement-method study. It is not a clinical prediction model, a claim
that the existing telecare measurements have been repaired, or evidence that
the proposed method is novel. Update the status and experiment log as work proceeds.

## 1. Scientific purpose

Determine whether cleaning or rejecting wearable heart-rate readings can remove
real within-person changes, and whether directly assessing uncertainty in those
changes improves monitoring at useful coverage.

The intended research users are developers and evaluators of remote-monitoring
systems. The immediate output is a supported increase/decrease/stable/uncertain
decision, evaluated against an independently acquired ECG reference. A heart-rate
change is not itself a diagnosis or an adverse clinical event.

Illustrative example: an observed rise from 75 to 105 beats/min may be a real rise
or an artifact. A smoother may suppress either. The question is whether a method
can preserve the reference-confirmed rise without also preserving false rises.
These values illustrate the problem; they are not clinical decision thresholds.

## 2. What changes from the earlier research

Earlier work in this repository studies alarm forecasting and prioritization.
Its outcome categories reflect operational records; they do not establish the
accuracy of a physiological measurement. Prior exploratory analyses did not
establish that selecting denser recordings resolves measurement validity or
produces a stable, distinctive personalization advantage. Previously inspected
validation/test records remain exposed.

The new study uses paired sensor/reference measurements, a change-inference
endpoint, and a separately frozen evaluation. Historical work and its limitations
remain part of the scientific record; do not copy its processed values, labels,
splits or normalization into this study without an explicit justification.
See the preserved [Study B overview](../../docs/studies/study-b.md) and
[public reproducibility guide](../../docs/REPRODUCIBILITY.md).

## 3. Questions and candidate contribution

**Q1 — establish the failure:** on naturally imperfect recordings, do methods
that improve pointwise heart-rate error also suppress, reverse or fabricate
reference-confirmed changes? Quantify how often, for whom and under which
recording conditions. A hypothetical failure is not enough.

**Q2 — test a remedy:** with the same underlying estimator and information,
does a model of change error outperform pointwise error filtering and a direct
change-regression baseline with ordinary calibrated intervals? Compare at useful
coverage and false-change burden, including real changes lost to abstention.

**Q3 — bounded portability analysis:** does that result survive a different
device/cohort, and how does it depend on receiving only scalar readings versus
additional signal-quality information? This is an ablation within the same study,
not a second unconstrained representation-learning project.

Possible technical work includes joint uncertainty for the preceding and recent
measurements, serially dependent errors, and personal reference uncertainty.
An offset shared by both periods cancels in their difference; changing bias does
not. Neither ordinary differencing nor this observation is a novelty claim.
Use the simplest method that addresses an empirically established failure.

The main scientific claim, if supported, would be that a particular error-control
strategy preserves more independently confirmed changes at a stated burden and
coverage across specified datasets. Do not promise guarantees under arbitrary
device shift or claim correction of unreferenced local physiology.

## 4. Constraints and data roles

- The core study must run with public reference datasets and existing compute.
  New recruitment, proprietary pretrained weights, paid datasets and company
  participation are not dependencies.
- The company may provide only another export with the same fields as the old
  November–January batch, and may participate only after a model is ready.
  Plan for that limited cooperation; do not assume ECG, new labels, calibration
  metadata or prospective measurement sessions will appear.
- A later unreferenced export can test schema compatibility, coverage, population
  and recording shift, explanation availability and computational operation.
  It cannot validate measurement accuracy, missed clinical events or benefit.
- The existing telecare data may inform acquisition-stress scenarios. Gaps do
  not establish non-wear; repeated values do not prove error. Clearly label
  simulated gaps and hypothetical bias separately from observed sensor errors.
- Disease/medication snapshots without timing cannot be prospective inputs.
  Dated context can describe cohorts or prespecified strata when permitted and
  sufficiently complete. Demographic or disease labels alone do not fix sensors.

Candidate selection starts with [EVIDENCE_AND_DATASETS.md](docs/EVIDENCE_AND_DATASETS.md).
One development resource and one locked external cohort are the default scope.
Prioritize an actual numerical wearable-output/ECG pairing for the scalar claim.
If only HR derived by our own PPG estimator is available, state that difference.
Choose datasets by reference, chronology, population and access suitability before
inspecting comparative performance. Do not choose the external set because it
gives the strongest result. Different versions of the same people are one cohort.

## 5. Protocol essentials

The full protocol is [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
Before final fitting, freeze the following items in a versioned configuration:

1. Population and exclusions, data versions, participant grouping and reference
   quality rules; the latter must not depend on a candidate model's predictions.
2. Decision time, preceding/recent intervals, nonoverlapping episode rule,
   reference alignment, causal availability and supported change time scale.
   Do not assume a six-hour task from short recordings.
3. Change magnitude/duration rule and how reference variability creates an
   indeterminate label. Synthetic-demo cutoffs are not study defaults.
4. Allowed scalar/history inputs, optional metadata, baseline definitions,
   fitting/calibration splits, tuning budget and threshold selection objective.
5. Primary operating point, practical improvement requirement, participant-level
   precision analysis, uncertainty method and subgroup-reporting limits.
6. External evaluation roster/access state, code/config hashes and the rule for
   documenting any post-freeze amendment or additional evaluation exposure.

Required comparisons include unchanged numerical estimates, last observation,
causal median/robust smoothing, direct change regression with calibrated intervals,
point-error filtering, and the closest applicable selective-correction method.
Keep the same information, training allowance and underlying estimator where
the comparison is intended to isolate the proposed component. Waveform-assisted
methods are an explicitly richer-information comparison.

Report point error and change error separately. Include wrong-direction and
false-change decisions, genuine changes suppressed by correction, changes lost
to abstention, reference-eligible sensitivity, population/time coverage,
calibration/interval width and runtime. Report selected-case performance alongside
all-reference-eligible denominators. Compare identical accepted sets as well as
matched coverage, so correction and selection contributions can be distinguished.

Use participant-grouped fitting/calibration and independent cohort evaluation.
Never let overlapping windows cross a split. Personal history is strictly prior
and allowed only as specified for the intended known/new-person use case.
Bootstrap participants or use a justified repeated-measures analysis; windows
are not independent subjects. Fixed thresholds transferred to a new dataset and
thresholds recalibrated with new reference labels are different experiments.

## 6. Milestones and stop conditions

| Milestone | Planned period | Required evidence / decision |
|---|---|---|
| M0: executable foundation | Initial branch setup | Synthetic tests, manifest verification, public instructions; no scientific claims |
| M1: data and novelty gate | First half of November | Reference/timing/license audit; fair simple baselines; documented real failure and literature gap, or a stop/reframe decision |
| M2: focused development | Second half of November–early December | One justified intervention; internal comparisons; prespecified operating points and protocol freeze |
| M3: independent evaluation | December | Frozen evaluation on one independent cohort; participant-level uncertainty and coverage; no tuning on its outcomes |
| M4: explanation and reproducibility | January | Auditable outputs, tested explanation fidelity, complete scripts/artifact manifest and manuscript-ready figures/tables |
| Optional company evaluation | After a frozen package exists | One specified batch and report, with claims matched to the reference actually available |

M1 proceeds only when the selected datasets support the same defined question,
the failure is observed in natural sensor error, and the candidate contribution
is distinguishable from existing methods. A numerical practical-effect threshold
must be specified from intended use and reference precision before final testing;
it is not currently established.

Stop expanding this branch if ordinary baselines already solve the problem,
references cannot resolve the target change, or improvement depends on artificial
corruption alone or exclusion of most difficult people. Preserve a useful
benchmark/negative result instead of manufacturing a new architecture. If
measurement changes carry no testable distinction, discuss a new question before
starting an unrelated model search. A small exploratory effect is not sufficient
to justify repeated external tests.

## 7. Clear and testable explanations

Each future model output must state the observed baseline/recent summary,
estimated change and uncertainty, recording support, decision and reason for
indeterminacy. Distinguish uncertainty in the reference during evaluation from
uncertainty the deployed model can estimate. Do not use future reference values
inside model-side explanations.

Prefer directly auditable contributions and error analysis to a decorative
post-hoc explanation panel. If a learned component is used, validate explanation
stability and fidelity; feature importance is not a causal account. See
[EXPLANATIONS.md](docs/EXPLANATIONS.md) for concrete acceptance checks.

## 8. Reproducibility and boundaries

The package structure is described in [ARCHITECTURE.md](docs/ARCHITECTURE.md).
Run [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before adapting a dataset.
The initial test/demo implementation is intentionally small and uses no patient
data. A successful synthetic run does not complete M1.

Keep raw datasets, subject lists, detailed traces and generated runs in ignored
local locations. Dataset and code licenses are separate. Do not silently add a
license to third-party material or redistribute data because a paper is open.
Publish original scripts/configuration, aggregate results permitted for release,
source references and a replayable provenance record. Future dependency additions
need a versioned, tested environment; do not reuse the historical requirements as
an assumed model environment.

The [company request](docs/COMPANY_EVALUATION_REQUEST.md) is a preparation document,
not authorization to contact anyone. Company testing is initially offline/shadow
evaluation, without altering real alerts or care. No training claim, correction
claim or clinical claim should exceed the reference evidence obtained.

## 9. Current implementation and first next action

| Component | State at branch creation |
|---|---|
| Scientific question and evaluation design | Proposed; novelty and practical effect unverified |
| Public source/dataset shortlist | Documentation reviewed; recordings not downloaded or fitted |
| Synthetic contracts, metrics, manifest and regression tests | Initial implementation; exact checks recorded in the experiment log |
| Dataset adapters and reference quality audit | Not implemented |
| Trained error/change model and calibrated intervals | Not implemented |
| External evaluation lock and actual dataset split | Not created |
| Company model package or company evaluation | Not implemented or requested |
| Scientific performance results | None for this new study |

**First next action:** execute M1, beginning with source terms, raw/schema
inspection, reference timing and independent-participant inventory for the
preferred dataset pair. Create a development-only feasibility plan and exposure
register before running performance comparisons. Reserve the external cohort's
outcomes from method development. If data are unsuitable, record why before
considering a replacement. Do not start a large training sweep.

After each task, append to [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md), update the
status above, and record one bounded next action. Keep the main question focused
unless a documented feasibility result justifies a discussed change.
