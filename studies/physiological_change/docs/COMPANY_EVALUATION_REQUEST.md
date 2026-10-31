# Request for a later telecare evaluation batch

This is a technical request specification for a research discussion. It has not been sent, and company participation is not a prerequisite for model development. The core study uses independent reference datasets. The requested company contribution is one later export and, if convenient, a review of a frozen research prototype after development.

## Purpose and requested decision

We are investigating whether a method can interpret changes in wearable heart-rate readings while explicitly identifying insufficient recording support. We would like to test whether the completed method accepts the service's actual exports, how often it can produce a supported output, and how its explanations would fit the service's workflow.

**Minimum request:** provide one previously unseen, consecutive batch in the existing export format, together with a short field-definition sheet and known changes to devices or export processing. No new participant measurements, additional appointments, clinical annotations or changes to live alarm handling are required for this minimum contribution.

The study team will freeze the model and evaluation plan before opening the batch. The company need not help select or train the model. If a local evaluation is preferred, the team can prepare a frozen evaluation package after development, subject to the company's environment and data-handling arrangements. Such a package is planned; it is not currently a validated clinical tool.

## 1. Batch selection and timing

Agree on one complete observation interval and delivery date before transfer. Four consecutive weeks are a useful initial engineering evaluation; a longer routinely available export may be used if feasible. This duration is not a claim that the batch will contain enough clinical events for an outcome study.

An example schedule, if the model is frozen by 30 November 2026, is an export covering **1–31 December 2026**, delivered in early January 2027. These dates are a proposal to agree, not an established company commitment. If an already collected but previously unseen period is easier, record its dates and previous exposure explicitly. It can support frozen retrospective evaluation; it does not become prospective simply because it was delivered later.

Include all participants satisfying an agreed service or geographic criterion during the interval. Include sparse wearers, failed transmissions and ordinary operational difficulties where recorded. Do not select people based on previously good model results, high apparent measurement quality or absence of alarms. If only a subset is possible, agree a consecutive or reproducible sample before examining model outputs and provide the eligibility count and reasons for exclusions.

Use stable study pseudonyms where permitted so the research team can distinguish participants seen in the historical development data from new participants. Keep the identifying crosswalk with the company. Do not include names, addresses, contact details, national identifiers or unrestricted free-text notes in the minimum export.

## 2. Minimum deliverables

| Deliverable | Requested content | If unavailable |
|---|---|---|
| Measurement export | Existing original export format, participant pseudonym, measurement timestamp, channel/type, recorded value and units; preserve the original numeric representation | Provide the normal export unchanged and identify missing definitions. Do not create synthetic replacements. |
| Field-definition sheet | Meaning of each channel, units, missing-value markers, timestamp basis, and whether values are measured, estimated, averaged, accumulated or copied from another system | Mark each unresolved item as unknown. A guessed definition is less useful than an explicit unknown. |
| Export provenance | Export date, covered interval, source system/version if known, row counts and known filters, clipping, aggregation, deduplication or formatting changes | Describe the normal export steps in plain language if no formal specification exists. |
| Device/process change note | Device model and firmware where known; replacement dates, calibration requirements or processing changes during the batch | State what is known at service level; participant-level metadata is optional if not recorded. |
| Coverage description | Which service participants were eligible and any known enrollment, withdrawal, outage or device-replacement periods | State that enrollment/wear coverage is unavailable; the team will not infer continuous wear from first and last readings. |

For heart rate, specify whether the field is a sensor estimate, a displayed device value or a platform-derived summary, and any known averaging interval. For blood pressure, specify the source device or estimation pathway, units and any calibration information; the initial heart-rate evaluation does not require adding BP as an input. For steps, state whether the field is cumulative and how resets work. Other channels can remain in the ordinary batch without becoming primary research targets.

Preserve zeros, duplicate timestamps, missing fields and unusual values as exported; explain known special codes. Avoid additional cleaning specifically for this study. If the service already provides only processed data, identify that fact rather than reconstructing fictitious raw values.

Provide measurement time and, if already stored, receipt/availability time separately. State timezone and daylight-saving treatment. Do not infer arrival times from row order or export date. Without receipt times, the study will report measurement-time replay and cannot establish online timeliness.

## 3. Optional information already held by the company

These additions are useful only if they already exist and can be shared appropriately. They are not conditions for the minimum export:

- Device identifiers that permit within-study device changes to be tracked, plus battery, wear or signal-quality flags and transmission status.
- Dated calibration records, device replacements, enrollment or downtime intervals.
- Existing synchronized independent reference readings, with reference device, timestamp and measurement procedure. For heart rate, an ECG reference is preferable to another optical consumer estimate. Paired references would enable a separately defined accuracy evaluation.
- Existing structured alarm/action records, if a later operational study is desired: alarm origin and time, action time, action type, independent outcome source where available, and the meaning of each category. A standard operator template is not an adjudicated clinical outcome.
- Dated demographic or clinical context already available at the decision time. Undated disease and medication lists should be identified as snapshots and should not be supplied as longitudinal treatment history.

Do not collect extra sensitive information solely to fill these optional fields. If clinical outcomes or references are offered, the team will specify their evaluation and access arrangements before examining results.

## 4. Proposed execution and returned results

There are two possible arrangements: approved local analysis by the research team, or a company-side replay using a fixed package. Choose the one that fits existing infrastructure. In either case, retain the source files unchanged and calculate checksums. Do not transfer patient records through a public repository, ordinary issue comments or third-party model APIs.

The frozen package should identify the code revision, environment, input schema, preprocessing and fitted artifact checksums. It should run without automatic downloads, record incompatible fields rather than silently reinterpret them, and produce a private evaluation report. Hardware requirements and the exact invocation will be specified after implementation; no particular container platform is assumed now.

Replay is **shadow evaluation**: outputs are not delivered as clinical alerts, do not replace existing alarms, and do not alter care. The service continues its ordinary procedures.

With the minimum export, the returned report can contain:

1. Data-schema compatibility, timestamp/gap summaries and known processing discrepancies.
2. The proportion of decisions, person-time and participants for which the model has sufficient input support, including sparse and new participants.
3. The frequency and reasons for unsupported or rejected outputs, and differences from the public development data.
4. Execution time and resource use; observed transmission delay only if genuine availability timestamps exist.
5. Deidentified illustrative explanations, reviewed for disclosure risk before any wider sharing.

This batch alone cannot establish measurement accuracy, correct correction of values, clinical sensitivity/specificity, reduced workload or improved outcomes. A stable distribution is not evidence of accurate readings, and distribution change does not identify its clinical cause.

If an operator can provide feedback after the prototype is ready, ask whether the explanation is understandable, whether the reason for insufficient support is clear, and what information is missing for an actual decision. Record the review procedure and number of reviewers. Usability feedback is separate from physiological or clinical ground truth.

## 5. Independence and reporting

Keep the batch sealed from method selection until the model, thresholds, inclusion rules and report specification are frozen. Report prior/new participant strata and any device/process changes. If adaptation becomes necessary, preserve the original frozen result and label the adapted analysis as exploratory. Do not reuse the same batch as an untouched test afterward.

Only disclosure-reviewed aggregates and permitted examples are candidates for public release. The team should agree permitted uses, retention, access and publication review with the data holder through the existing institutional process. The scientific report will state what the batch can and cannot validate and will preserve negative or incompatible results.

## A short response that would make this request actionable

The company can answer the following without a technical meeting:

1. Is one consecutive, previously unshared batch in the existing format available, and for which dates?
2. Can a short field-definition/process-change sheet accompany it, with unknown items marked explicitly?
3. Can stable research pseudonyms and basic cohort eligibility information be retained?
4. Is approved research-team analysis or company-side replay preferred?
5. Are any independent reference readings or richer structured operational records already held, without additional collection?

These answers determine the evaluation scope. A new batch without reference measurements is still useful for compatibility and coverage testing; it does not become a measurement-accuracy validation.
