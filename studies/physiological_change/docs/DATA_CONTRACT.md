# Data and timing contract

This is the intended dataset-adapter contract, not a statement that all these
fields exist in the current telecare exports. The initial synthetic dataclasses
implement a smaller subset. An adapter must document missing fields and preserve
the meaning of its source before claiming conformity.

## Source registration

For every source, record dataset/version URL, retrieval date, license and access
terms, file hashes, cohort identifier, device/firmware if known, population,
protocol, sensor modalities and reference method. Record prior use of that cohort
by this project, overlap with other releases and any exposure of evaluation
outcomes. Dataset metadata inspection and outcome-based model development are
different exposure events; record both accurately.

Dataset files are immutable. A new parser or timing correction creates a new
adapter version and derived artifact, never a silent change to the raw source.

## Canonical observations

| Field / concept | Requirement |
|---|---|
| Dataset, participant, session and device keys | Namespaced; pseudonymous; stable grouping across overlapping source releases |
| Measurement timestamp or source-relative offset | Explicit unit and origin; UTC/offset or documented relative clock; never infer ns/us/ms from magnitude alone |
| Availability / receipt timestamp | Separate if supplied; unknown must remain unknown |
| Numerical HR value and unit | Beats/min after a documented conversion; missing is not zero; preserve raw representation |
| Source record and derivation | Trace value to original file/record and estimator version; flag directly exported HR versus HR computed from PPG |
| Quality / activity / device context | Keep original definitions; no inferred non-wear label from a gap alone |
| Duplicate/carry-forward status | Preserve detection rule and conflicting values; do not assume repeated values are artifacts |
| Reference ECG/HR and timing | Separate stream with alignment method, quality and uncertainty; never exposed to inference by default |

Do not make future-informed alignment a hidden preprocessing advantage. If a
public source was aligned using its entire reference recording, declare that
retrospective alignment and avoid an online synchronization claim. Record any
reference-driven filtering of physiological extremes; those excluded regions
cannot be used to support claims about abnormal physiology.

## Episode contract

An episode has one decision time, a preceding baseline interval and a recent
interval. Their boundaries, aggregation method, required support and maximum
permitted gaps must be declared before evaluation. Model-side inputs end no later
than decision time and must respect known availability. Future reference values
may define an offline target only when that target/time window is explicitly
part of the protocol; they cannot leak into inference or explanations.

For the primary same-time change study, align both reference summaries to the
same preceding/recent periods. Record the uncertainty of each and of their
difference, rather than treating noisy reference means as exact truth. Keep:

- reference unavailable;
- reference available but unable to resolve the change category;
- reference eligible and the model abstains;
- reference eligible and the model makes a decision.

These states have different denominators. A model abstention cannot become a
reference exclusion. Loss of reference during motion can select the population;
report that loss before measuring model coverage.

The episode identifier must be deterministic from namespaced source/session and
interval definitions. Avoid overlapping primary episodes within a participant;
if a sensitivity uses overlapping windows, disclose dependence and do not count
them as independent outcomes.

## Splits, fitting and evaluation

Split people (including all device sites and sessions) together for new-person
evaluation. Preserve chronology for any within-person future-use experiment.
All fitted imputation, normalization, calibration, feature selection and tuning
must respect those roles. A previous reference reading may be a model input only
in an explicitly budgeted reference-assisted experiment, not the scalar-only task.

Store source/group membership locally and publish only appropriate aggregate
counts and split/protocol hashes. Freeze the external roster and method before
scoring. A protocol amendment must record what had already been observed.

## Later company exports

Use the same checks, but do not invent a reference column. Missing device or
availability information narrows claims. Keep existing-person and new-person
groups distinct when stable pseudonyms permit linkage. Without stable linkage,
new-person independence cannot be established. See the exact
[company request](COMPANY_EVALUATION_REQUEST.md).
