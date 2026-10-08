# Research overview

The [data pipeline](PIPELINE.md) is the shared engineering foundation: ingestion,
measurement validity, causal features, analytical storage and task-specific datasets.
Its architecture and evaluation are central to the [processing-pipeline thesis](THESIS.md).

AI-CVD investigates longitudinal wearable measurements in older-adult telecare.
The name is a project identifier; it does not imply cardiovascular diagnoses or a
validated cardiovascular-event detector. Physiological channels and activity are
sparse, irregular and affected by how measurements are acquired.

| Work | Prediction setting | Interpretation |
| --- | --- | --- |
| AIME 2026 | Unsupervised reconstruction/anomaly scoring in a feasibility study | Published baseline; not a production detector |
| Study A | Continuous prediction times before a subsequent recorded alarm | Repaired early-warning task; internally completed |
| Study B | A recorded SOS episode is already the anchor | Prioritization of subsequently recorded telecare outcomes |

Continuous early warning asks **whether an event will occur in a future horizon**.
Alarm-conditioned prediction asks **which recorded outcome is associated with an
already initiated alarm**, using only history available beforehand. Conditioning
on SOS changes the population and use case.

Later work repaired the temporal/task definitions and introduced explicit
missingness, causal history, subject isolation and frozen evaluation protocols.
Earlier exploratory CGTA/fusion results used a different task. The historical
AUROC 0.782 is not evidence of a validated improvement over AIME or the later studies.
Corrections and failed experiments remain in the preserved research record;
a cleaner public tree is not a rewrite of scientific history.

The published source is described in [Publications](PUBLICATIONS.md). Internal-study
status and [limitations](LIMITATIONS.md) remain explicit. Public implementation scope
is indexed in the [code map](architecture/code-map.md).
