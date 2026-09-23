# Canonical feature dictionary — schema v2

Authority: `configs/tasks/level3_4h.toml`; implementation:
`src/ai_cvd/features.py`. This supersedes the historical Parquet dictionary.
The raw database is read-only. Features are generated independently of alerts.

## Grain, timing and inputs

One row is one patient and one five-minute bucket. `timestamp` is its inclusive
start; `available_at` is its exclusive end. The row is unavailable before that
end. A sample at t contains 96 rows with starts in `[t-8h,t)` and availability
no later than t. Source readings at exactly t never enter X_t. The primary target
is a Level 3 recorded escalation in `(t,t+4h]`, assigned from episodes separately
in the sample manifest. Level 2+3 is an explicitly selected secondary endpoint.

`senior_id`, timestamps, sample IDs, labels, outcomes, episode IDs, lead times,
split and task IDs are metadata, never model inputs. `FEATURE_NAMES` is the fixed
ordered allowlist, copied into every run and exported array. No dynamic column
selection, normalizer, imputation or neural architecture change occurs here.

## Raw validity, duplicates and physiology

Validity bounds are inclusive and live in the canonical task config. Nonfinite,
unparseable and out-of-range values become NULL; they are not clipped to plausible
extremes. Exact patient/time/type duplicates are consolidated before bucketing.
For physiological values the mean of valid duplicates is used. Conflicting step
values at exactly the same time are rejected. Every invalid channel remains missing
and cannot reset its valid-observation timer. Inverted BP pairs nullify both values.

| Feature | Units | Definition |
|---|---|---|
| `temperature` | degrees C | Mean valid bucket readings, inclusive validity 30–45 |
| `heartrate` | bpm | Mean valid bucket readings, 30–220 |
| `sbp` | mmHg | Mean valid systolic readings, 70–250 |
| `dbp` | mmHg | Mean valid diastolic readings, 40–140 |
| `saturation` | percent | Mean valid SpO2, 50–100; both tails invalidated |
| `steps` | count, subject to explicit source semantics | Sum of known increments ending in this bucket; missing if no computable increment |
| `pulse_pressure` | mmHg | Mean valid raw paired SBP−DBP, first within duplicate groups then within buckets; positive differences below configured 10 mmHg minimum are missing |
| `shock_index` | ratio | Bucket mean HR / bucket mean **SBP**, only when both valid |

Pulse pressure has one authoritative function, `pulse_pressure()`. It is calculated
from valid paired readings, never resurrected from independent bucket means after
being invalidated. Valid SBP/DBP can remain available when a narrow PP is missing.
If only one BP component is recorded/valid, that component can be observed but PP
is missing. Shock index uses co-bucket measurements, not necessarily simultaneous
measurements; synchronization within five minutes is an explicit approximation.

## Steps — do not assume increments

The source contract must declare and justify `increments` or `cumulative_counter`.
Unknown semantics cause an error. There is no default sum of private step readings.

For increments, exact duplicates count once and distinct observed increments sum.
For cumulative counters, differences are taken only between successive valid
readings. The first reading and any decrease/reset give unknown activity; the new
counter initializes subsequent differences. No activity is imputed across a reset.
The source contract also requires `counter_reset_policy`: `daily` suppresses every
cross-date delta in the declared source timezone, even without a visible decrease;
`decrease_only` is an explicit alternative assumption for non-daily counters.
The private source audit strongly supports daily cumulative counters empirically;
it is not authoritative device documentation. Reset timezone remains a declaration.
Known zero differences are observed zero activity. A delta is attributed to the
bucket containing its endpoint and can span more than five minutes; the interval
duration below makes this visible. Negative/raw values above the configured limit
are invalid; a device with a different range needs a separately versioned contract.

| Additional feature | Definition |
|---|---|
| `steps_source_value` | Last valid original step reading in the bucket, before differencing |
| `observed_steps_source` | 1 if at least one valid original step reading exists, including initial/reset counters |
| `steps_counter_reset` | 1 if a valid cumulative reading decreases, or crosses a local date under the daily reset policy |
| `steps_delta_interval_minutes` | Maximum interval spanned by computed cumulative deltas in the bucket; NULL for increments/no delta |

## Observation masks and recency

For each of `temperature`, `heartrate`, `sbp`, `dbp`, `saturation`, `steps` and
`pulse_pressure`, `observed_<name>` is 1 if a valid bucket value exists, else 0.
`time_since_last_<name>` is elapsed minutes from the latest valid contributing
reading to bucket **end**. It is NULL before the first valid observation and
otherwise nonnegative. Invalid/NULL readings never reset it. BP and PP each track
their own valid observations, rather than blindly sharing a source-record timer.
For steps the timer/mask concerns computable activity increments; source-counter
observation is separately represented above.

Missing steps are NULL with mask 0; observed zero steps are 0 with mask 1. Missing
vitals are never mean-filled by this pipeline. Any downstream model must preserve
the masks and fit preprocessing only on training data.

## Causal derived and clock features

| Feature | Definition |
|---|---|
| `hr_bucket_sd_4h` | Sample SD of observed bucket-mean HR over the trailing four hours, at least two observations |
| `bp_trend_mmhg_per_hour` | SBP least-squares slope over trailing three hours, expressed per hour; NULL with fewer than two observations |
| `steps_sum_6h` | Sum of known step increments whose endpoints fall in the trailing six hours; NULL with no observed increments |
| `steps_observed_buckets_6h` | Number of buckets contributing a known step increment to that sum |
| `hour_sin`, `hour_cos` | Sine/cosine of UTC hour including minutes at bucket start |
| `is_night` | 1 for UTC clock hour [00:00,06:00); not a measured sleep state |

Trailing intervals include the current closed bucket and exclude buckets starting
before `available_at - rolling_duration`. Partial activity coverage is not an exact
six-hour activity total. Cumulative deltas can span outside that trailing interval;
the duration feature and masks must inform interpretation. Recency missingness is
not automatically illness. `hr_bucket_sd_4h` is **not beat-to-beat HRV** and must not
be described as such. No RR intervals are available to establish beat-level HRV.

## Clinical snapshots

Optional numerical fields are `age`, `gender`, `cardiovascular`,
`metabolic_endocrine`, `neurological`, `psychiatric_cognitive`, `musculoskeletal`,
`respiratory`, `gastro_renal_urologic`, `oncological`, `sensory`,
`other_functional_risk`, and `other`. Each has a `known_<name>` mask.

Age must be age at the documented snapshot, not recalculated using the current
calendar year. Gender coding, if used, must be declared by the snapshot producer;
the established convention is 0 male / 1 female, NULL unknown. Disease domains
are multi-hot indicators, not mutually exclusive one-hot classes. Unknown history
is NULL, not absence of disease. Undated current disease/medication tables are not
automatically eligible inputs. Only snapshots available strictly before bucket end
can contribute. `verified_baseline` requires a provenance declaration and dates no
later than enrollment; otherwise use `dated_snapshots` or `exclude`.

## Deliberate changes from historical schema

Removed future labels from feature artifacts, `recent_event_burden` pending as-of
validation, raw `hour`/`day_of_week`, legacy `hr_volatility`, and ambiguous activity
summation. Added explicit availability, masks, known-history flags and source-step
diagnostics. New names, ordering, schema version and task fingerprints deliberately
prevent reuse of historical arrays/checkpoints without a future migration task.

## Exact array feature order

<!-- FEATURE_ORDER_START -->
```text
temperature
heartrate
sbp
dbp
saturation
steps
pulse_pressure
steps_source_value
observed_steps_source
steps_counter_reset
steps_delta_interval_minutes
shock_index
hr_bucket_sd_4h
bp_trend_mmhg_per_hour
steps_sum_6h
steps_observed_buckets_6h
hour_sin
hour_cos
is_night
observed_temperature
observed_heartrate
observed_sbp
observed_dbp
observed_saturation
observed_steps
observed_pulse_pressure
time_since_last_temperature
time_since_last_heartrate
time_since_last_sbp
time_since_last_dbp
time_since_last_saturation
time_since_last_steps
time_since_last_pulse_pressure
age
gender
cardiovascular
metabolic_endocrine
neurological
psychiatric_cognitive
musculoskeletal
respiratory
gastro_renal_urologic
oncological
sensory
other_functional_risk
other
known_age
known_gender
known_cardiovascular
known_metabolic_endocrine
known_neurological
known_psychiatric_cognitive
known_musculoskeletal
known_respiratory
known_gastro_renal_urologic
known_oncological
known_sensory
known_other_functional_risk
known_other
```
<!-- FEATURE_ORDER_END -->
