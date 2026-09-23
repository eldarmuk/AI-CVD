# AI-CVD primary scientific protocol — version 2

## Authority and status

`configs/tasks/level3_4h.toml` is the sole task configuration. The loader validates
its contracts and embeds a SHA-256 fingerprint in every sample and run. This
protocol supersedes historical experiments; it does not validate their results.
No models were retrained during the data-pipeline repair.

The supported entry point is `python -m src.ai_cvd.cli`. The numbered data scripts
now delegate to it. The old data implementations are preserved under
`src/archive/legacy_v1/`. Historical model/evaluation entry points refuse direct
execution: their thresholding, feature schemas and artifact formats have not yet
been migrated. They must not consume v2 data or be reported as the primary task.

## Task, intervals and endpoint

| Item | Primary definition |
|---|---|
| Grid | 5 minutes, UTC |
| Sequence | 96 contiguous buckets |
| Lookback | 480 minutes = 8 hours, never 24 hours |
| Prediction cadence | Every grid boundary |
| Input | Bucket starts in `[t - 480 minutes, t)` |
| Forecast | `(t, t + 240 minutes]` |
| Primary endpoint | First recorded Level 3 alert in an episode |
| Optional secondary | First recorded Level 2 or 3 alert in an episode |
| Negative | No qualifying recorded event in the horizon, with verified coverage |
| Split unit | Patient, before eligibility, windowing or sampling |

Level 3 is a documented emergency-escalation **proxy**, not an adjudicated diagnosis
or a proven cardiovascular crisis. Level 2 is failed contact, not proven clinical
deterioration. The keyword classifier is versioned code and requires independent
clinical validation. Unknown notes are not proof of health: the estimand is
recorded escalation under the ascertainment declaration, not all biological events.

The target event time is the first constituent alert with a qualifying severity.
Raw notes have no separate severity-recording timestamp in the inspected schema.
Private extraction therefore requires `alert_time_semantics` and `alert_time_evidence`
declaring that severity was recorded at the alert timestamp, or identifying this as
an explicit retrospective timing assumption. Later-edited notes cannot establish
escalation timing without that assumption or additional source evidence.
For a Level 1 alert at 11:55 and Level 3 at 12:00, the primary event time is 12:00.
It is not backdated to 11:55. This choice avoids using a later escalation outside
the prediction horizon to redefine an earlier low-severity alert as a crisis.
The secondary target must be requested with `--endpoint level2_or_level3`; it
uses separate run directories and explicit endpoint metadata. Never merge endpoint
results in one comparison table without identification.

At an isolated event time E, the positive grid-aligned prediction times are
`E - H, E - H + grid, ..., E - grid`: 48 times for the primary horizon. A prediction
at E is negative for that episode because the left target boundary is open.
The event at exactly `t + H` is positive. All matched episode IDs are preserved;
`episode_id` and `lead_minutes` refer to the earliest qualifying future episode.

## Buckets and feature availability

A row's `timestamp` is its bucket **start**, not its availability time. It covers
`[timestamp, timestamp + grid)` and has a separate `available_at` equal to the
exclusive end. At prediction t, all 96 rows have `timestamp < t` and
`available_at <= t`; a raw measurement at t is excluded. A measurement at 10:04
belongs to 10:00–10:05, becomes usable at 10:05, and has recency one minute then.

Source acquisition time alone does not establish delivery time. The source
contract must either supply `explicit_available_at` or explicitly declare the
assumption `timestamp_is_available_at` with evidence. Late readings are
conservatively omitted from the closed bucket, never retrospectively backfilled.
Naive timestamps require a declared source timezone. Ambiguous or nonexistent DST
timestamps must be resolved upstream; the adapter rejects them rather than guess.
Clock features are explicitly UTC clock time, not an inferred patient sleep phase.

Rolling descriptors use only preceding/current closed buckets. Their lookback
may extend before the first bucket in a sequence; that is a documented causal
feature warm-up, not future information. Before the run-in is complete, no sample
is eligible. Feature order is the fixed `FEATURE_NAMES` allowlist and is recorded
in every run and array shard; future labels, identifiers and episode fields are
never features. Static snapshots must have `available_at < bucket_end`.

## Episodes and available versus retrospective information

Adjacent alerts with a gap **at most 10 minutes** form a chained episode. The
threshold is inclusive; e.g. 00:00, 00:10, 00:20 are one episode. A gap greater
than 10 minutes starts another. Source alert IDs must be unique per patient.

Episodes preserve patient/episode ID, start/end, every constituent alert ID/time/
severity, first severity, maximum recorded severity, final recorded severity,
first escalation time, first time maximum severity was recorded, and member count.
Maximum recorded severity is not labelled clinician adjudication. A later
downgrade cannot erase an earlier Level 3 record. Episode IDs are deterministic
from patient, first alert ID and start; a changed source snapshot can change
episode boundaries and must be a different versioned run.

`alert_episodes_v2` and `alert_episode_members_v2` are new derived tables. The raw
database and historical processed tables are never altered. Event targets consume
episodes. Episode-wide maximum/final severity are **retrospective target/audit
information**, never model inputs. `recent_event_burden` is intentionally absent
from the v2 feature allowlist until an as-of severity implementation is validated.

## Causal cohort and ascertainment

Each patient needs an external coverage declaration:

```text
senior_id,enrollment_time,measurement_coverage_end,outcome_coverage_start,outcome_coverage_end
```

All timestamps must be timezone-aware. These are continuous ascertainment
intervals, not first/last observed reading times. In particular, a final measurement
does not establish discharge, and absence of a subsequent alert does not establish
negative follow-up. Measurement records outside the declared interval are excluded.

The cohort uses a fixed 24-hour run-in from declared enrollment, requiring at least
four valid HR buckets and two valid paired-BP buckets within that run-in. These
BP buckets require at least one valid paired pulse pressure under the configured
filter; independent unpaired SBP and DBP observations cannot satisfy this criterion.
These are **observed bucket counts**, not raw packet counts. The definitions and thresholds
live in the task file. Eligibility becomes known only after run-in completion and
never depends on subsequent monitoring density or lifetime alert status.

A sample also requires complete contiguous input history, outcome ascertainment
already active at t, and declared measurement/outcome coverage through `t + H`.
Insufficient follow-up is censored for both positives and negatives; it never
becomes an automatic negative. This is explicit retrospective ascertainment
restriction, not an input feature or a prospective enrollment rule.

Undated disease/medication tables do not establish baseline availability. Supply
dated numeric snapshots, explicitly attest verified baseline availability with
provenance, or choose `clinical_history_policy = "exclude"`. Exclusion produces
NULL static values with zero known masks, not false negative disease flags.
The primary runner does not silently join the existing undated risk-profile table.

The inspected schema lacks authoritative enrollment/discharge, ascertainment and
clinical-history availability fields. Until the data owner supplies those or a
separately documented study assumption is approved, private clinical eligibility
cannot be certified. The runner fails closed rather than manufacturing dates.

## Splits and manifests

A seeded SHA-256 patient assignment gives nominal 70/15/15 train/validation/test
fractions. Realized counts need not equal those ratios exactly. It is independent
of outcomes, record order and eligible windows, stable when other patients are
added, and does not reserve a selected count of future event patients. Any change
to the seed or split policy changes the task fingerprint and constitutes a new
experiment. This does not erase prior researcher exposure to the historical test
patients: a scientifically untouched final validation cohort remains necessary.

`stream_manifest.jsonl` contains **all eligible prediction times in every split**;
its test partition is the natural-prevalence held-out evaluation stream. It is
not balanced. Each row includes immutable SHA-256 `sample_id` from patient and
UTC prediction time, patient, prediction time, input bounds, target, linked episode
IDs, lead time, split, endpoint, feature version and full task identifier. A sample
ID names a patient/time pair; join using `(task_identifier, endpoint, sample_id)`
when mixing tasks/endpoints. Sample IDs are pseudonyms, not anonymization.

`development_manifest.jsonl` is optional, balanced by label within train and
validation only, and labelled **not natural prevalence**. Test examples are never
selected or balanced by this command. Its own provenance points to the original
stream checksum. No thresholding/calibration/model selection exists in these
data commands.

`unsupervised_training_manifest.jsonl` is a separate, explicitly retrospective
training-only subset. Default `event_free_windows` excludes actual recorded
severity 1/2/3 alerts throughout `[t - L, t + H]` and requires outcome coverage
from `t - L`. It does not inspect future label columns inside X. Optional
`never_event_patients` excludes patients with any such event in the supplied
snapshot. This is an outcome-selected reference cohort, not proof of health.
Neither policy changes stream eligibility or held-out evaluation prevalence.

`cohort_flow.json` counts source patients, missing-coverage patients, eligible
patients/prediction times, sequential disjoint window exclusions, retrospective
training windows, and target episodes with/without an eligible warning. Event loss
reasons and patient exclusion reasons can overlap; unassessable event opportunities
for patients missing coverage are explicitly not estimated. Zero-valued criteria
may be absent from the counter and mean zero, not missing ascertainment.

## Source contracts and step interpretation

`configs/source_contract.example.toml` is deliberately non-runnable until its
unknown declarations are replaced. No default converts private step values into
increments. A separate read-only source audit summarizes within-day trajectories,
repetition, cross-day resets, duplicates and sum/max behavior. Its local outputs
are private data-derived artifacts and must not be committed.

For verified **increments**, valid exact-time duplicates are counted once and
increments are summed. Conflicting step values at exactly the same timestamp are
rejected. For declared **cumulative counters**, only increases from the previous
valid reading become step deltas. The first reading and a decreasing/reset reading
give unknown activity, not zero. Their source value and source-observed/reset flags
remain visible. Delta-interval duration is retained because a delta ending in a
five-minute bucket may span longer than five minutes. The raw source values are
never naively summed. Unsupported semantics fail before extraction.

The completed local audit provides strong empirical evidence for daily cumulative
counters, not authoritative device documentation. The private aggregate audit is
retained locally and excluded from Git. Declare `counter_reset_policy = "daily"`
and the reset timezone only as documented source assumptions. Daily policy makes
every cross-date difference unknown, including invisible resets where the next
reading exceeds the prior day's final counter. `decrease_only` is supported only
as an explicit alternative declaration. Invalid readings never reset the counter.

## Artifacts and reproducibility

Each run requires a new output directory. Old arrays, metrics, checkpoints and
reports are never overwritten or silently reused. A complete run has task/schema,
source contract, source snapshot ID, runtime and artifact checksums. Partial runs
have no complete metadata and are rejected by consumers. The source snapshot ID
is an externally supplied immutable dataset identifier; the caller must ensure
the source snapshot stays frozen for the run (especially for the optional DuckDB
adapter). SQLite uses a read-only transaction. Private coverage/history manifests
must be preserved with the run for audit under the NDA.

Sequence export writes bounded NPZ shards with X, y, ordered feature names, task
identifier and sample IDs, plus row-to-sample mapping. It performs no imputation,
normalization, fitting or model training. Downstream predictions/embeddings must
use strict one-to-one ID joins, never concatenate based only on row count.
The join helper requires source-manifest checksum provenance on predictions and
the expected checksum, and rejects revised-source or different-manifest joins.
`arrays.verified_shards` checks shard and mapping checksums, per-row input hashes,
immutable IDs, target alignment, task, endpoint, split, dimensions and feature order.
Consume this iterator completely to validate the entire export. Keep predictions
bound to the source manifest checksum as well as task/endpoint/sample IDs: IDs alone
do not distinguish revised source snapshots at the same patient/time.

For public reproduction, the six fictional patients in `src.ai_cvd.synthetic`
have irregular readings, missing channels, zeros, duplicates, invalid values,
escalations, exact-boundary events, dated histories and censoring. No fixture
parameters were fitted to private data. Run the tests and synthetic commands in
the README. Production performance on the 200M+ raw-row dataset has not been
benchmarked: adapters push patient selection/order to the database and bound
feature processing to one patient, prioritizing a single verified implementation.
