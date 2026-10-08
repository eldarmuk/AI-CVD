# Source-neutral input contract

The public adapter reads local UTF-8 CSV files. An institution's export adapter maps
its authorized source into this schema; private database credentials, tables, labels
and notes are not inferred or embedded. No clinical free-text input is accepted.

| File | Required fields |
| --- | --- |
| telemetry.csv | record_id, subject_id, timestamp, available_at, measurement, value, sbp, dbp, unit, device_status |
| coverage.csv | subject_id, observation_start, observation_end, outcome_start, outcome_end |
| events.csv | subject_id, event_id, timestamp, outcome_code |
| source.json | synthetic flag and explicit contract declarations |

Generate a complete fictional example with
[fixture.py](../../src/ai_cvd/pipeline/fixture.py). Its source.json demonstrates every
required declaration without real institution-specific evidence.

Timestamps must be ISO 8601 with a timezone. Processing uses UTC. Local calendar-day
step-counter resets use the explicitly declared source timezone. Units must already
be normalized: Heartrate/bpm, Temperature/degC, BloodPressure/mmHg, Saturation/% and
Steps/count. Unsupported units are quarantined rather than converted by assumption.
BloodPressure uses sbp/dbp from the same paired observation; other types use value.

device_status accepts ok, off_body or device_error. These are canonical engineering
flags supplied by an upstream adapter, not a newly validated hardware-noise detector.
The code also masks out-of-range values and invalid pressure pairs. It does not claim
that all in-range sensor artifacts can be recognized.

record_id must be unique. Repeated observations can have different record IDs and are
handled by the feature aggregation rules. Reusing a record ID across different files
fails intentionally. Coverage has one row per subject and must be externally justified.
Outcome codes are explicit integers 0–3; their clinical interpretation is not inferred.

The contract requires steps_semantics, steps_evidence, measurement_availability,
availability_evidence, coverage_evidence, clinical_history_policy, alert_time_semantics
and alert_time_evidence. Cumulative counters also require counter_reset_policy and a
source timezone. The public orchestrator requires clinical_history_policy=exclude.
The retained feature function supports dated snapshots, but that adapter is outside
the advertised public workflow.

For authorized external data, set synthetic=false. `run` performs processing only;
the public model demonstration rejects such runs. Keep every generated source/data,
database, array and report in appropriately protected storage. Ignore rules are not
access control. Only generated fictional files are used in repository tests and CI.
