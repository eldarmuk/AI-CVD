# Feature Dictionary and System Mapping

This document defines the columns produced in `data/processed/multimodal_features.parquet`
by `src/pipelines/00_build_multimodal_features_duckdb.py` and consumed by the
sequence and benchmark pipelines. Rows are keyed by `senior_id` and a 5-minute
bucketed `timestamp`. Targets are future-looking labels over the next 24 hours.

## Dataset Grain and Split Policy

- **Row grain:** one senior, one 5-minute time bucket.
- **Window grain:** downstream sequence models use 96 consecutive rows, equal to
  a 24-hour lookback window.
- **Split unit:** `senior_id`. A senior must appear in exactly one of train,
  validation, or test.
- **Training anomaly policy:** unsupervised/anomaly training data is restricted
  to pure-healthy windows. For the Level 3 task, pure-healthy means no
  `label_1`, `label_2`, or `label_3` inside the lookback window.
- **Primary binary benchmark target:** `label_3`, indicating an acute crisis in
  the future 24-hour lookahead window.

## Raw Dynamic Physiological Signals

| Column | Unit | Type | Source and processing | Modeling role |
| --- | --- | --- | --- | --- |
| `temperature` | degrees C | float | Average temperature measurement in the 5-minute bucket. Missing when no temperature measurement was observed. | Core time-series physiology. |
| `heartrate` | bpm | float | Average heart-rate measurement in the bucket. Missing when not observed. | Core cardiovascular time-series physiology. |
| `sbp` | mmHg | float | Average systolic blood pressure from blood-pressure records in the bucket. Missing when not observed. | Core hemodynamic signal. |
| `dbp` | mmHg | float | Average diastolic blood pressure from blood-pressure records in the bucket. Missing when not observed. | Core hemodynamic signal and denominator for `shock_index`. |
| `saturation` | percent SpO2 | float | Average saturation measurement in the bucket. Missing when not observed. | Core respiratory/oxygenation signal. |
| `steps` | count per 5-minute bucket | float | Sum of step measurements in the bucket, with absent step observations coalesced to zero. Sequence generation caps extreme values at 2000. | Mobility and behavior signal. |

## Derived Dynamic Physiological Signals

| Column | Unit | Type | Definition | Notes |
| --- | --- | --- | --- | --- |
| `pulse_pressure` | mmHg | float | `sbp - dbp`. | Clipped to NULL when the bucket average has `dbp >= sbp`, preventing inverted pressure values from entering the model as valid physiology. |
| `shock_index` | ratio | float | `heartrate / dbp`. | Computed only when both `heartrate` and nonzero `dbp` are available. Used as a hemodynamic instability indicator. |
| `hr_volatility` | bpm | float | Four-hour rolling sample standard deviation of `heartrate`, partitioned by `senior_id` and ordered by `timestamp`. | Captures short-horizon autonomic/heart-rate variability. |
| `bp_trend` | mmHg per epoch-second | float | Three-hour rolling linear-regression slope of `sbp` against bucket time. | Captures directional SBP trajectory. Sequence generation fills missing values with `0.0`. |

## Circadian and Temporal Parameters

| Column | Unit | Type | Definition | Notes |
| --- | --- | --- | --- | --- |
| `timestamp` | datetime | timestamp | Five-minute bucket timestamp generated from each senior's observed measurement bounds. | Ordering key for all rolling features and sequence windows. |
| `hour` | 0-23 | integer | Hour extracted from `timestamp`. | Temporal descriptor. Excluded from neural input by the current shared feature selector. |
| `is_night` | binary | integer | `1` when `hour` is 0 through 5, else `0`. | Circadian/nighttime indicator. Excluded from neural input by the current shared feature selector. |
| `day_of_week` | 0-6 | integer | Day of week extracted from `timestamp`. | DuckDB `extract(dow)` convention. Excluded from neural input by the current shared feature selector. |
| `hour_sin` | cyclical encoding | float | `sin(2 * pi * hour / 24)`. | Continuous cyclical representation of hour. |
| `hour_cos` | cyclical encoding | float | `cos(2 * pi * hour / 24)`. | Complements `hour_sin` to avoid artificial midnight discontinuity. |
| `steps_rolling_sum_6h` | steps | float | Six-hour rolling sum of bucketed `steps`, partitioned by `senior_id`. | Recent mobility/activity load. |

## Behavioral Sparsity Parameters

Silence-tracking features measure elapsed minutes since the most recent observed
measurement of each modality for the same senior. They encode missingness and
care-interaction patterns without forward-filling the vital sign itself.

| Column | Unit | Type | Definition |
| --- | --- | --- | --- |
| `time_since_last_temperature` | minutes | float | Minutes since latest observed temperature measurement at or before `timestamp`. |
| `time_since_last_heartrate` | minutes | float | Minutes since latest observed heart-rate measurement at or before `timestamp`. |
| `time_since_last_sbp` | minutes | float | Minutes since latest observed blood-pressure measurement at or before `timestamp`. |
| `time_since_last_dbp` | minutes | float | Same blood-pressure recency timer as `time_since_last_sbp`. |
| `time_since_last_pulse_pressure` | minutes | float | Same blood-pressure recency timer as `time_since_last_sbp`; pulse pressure is derived from blood pressure. |
| `time_since_last_steps` | minutes | float | Minutes since latest observed step measurement at or before `timestamp`. |
| `time_since_last_saturation` | minutes | float | Minutes since latest observed saturation measurement at or before `timestamp`. |

## Static Demographics and Historical Context

| Column | Unit | Type | Definition | Notes |
| --- | --- | --- | --- | --- |
| `senior_id` | identifier | integer/string-like | Senior identifier. | Used for subject-wise isolation across splits. It must not be used as a predictive feature. |
| `age` | years | integer | Age from the senior demographic table. | Static demographic feature. |
| `gender` | binary | integer | Encoded as `0 = Male`, `1 = Female`; NULL when unknown or unmapped. | Static demographic feature. |
| `recent_event_burden` | count | float | Count of compressed alerts with severity 1, 2, or 3 in the preceding 48 hours, from `t - 48h` inclusive to `t` exclusive. | Historical context feature. It uses past alerts only and excludes current/future alerts. |

## Static Clinical Comorbidity Domains

The pipeline maps each senior's clinical history into one-hot disease-domain
indicators and joins the vector to every timestamp for that senior.

| Column | Type | Definition |
| --- | --- | --- |
| `cardiovascular` | binary | Cardiovascular disease-domain indicator. |
| `metabolic_endocrine` | binary | Metabolic and endocrine disease-domain indicator. |
| `neurological` | binary | Neurological disease-domain indicator. |
| `psychiatric_cognitive` | binary | Psychiatric or cognitive disease-domain indicator. |
| `musculoskeletal` | binary | Musculoskeletal disease-domain indicator. |
| `respiratory` | binary | Respiratory disease-domain indicator. |
| `gastro_renal_urologic` | binary | Gastrointestinal, renal, or urologic disease-domain indicator. |
| `oncological` | binary | Oncological disease-domain indicator. |
| `sensory` | binary | Sensory impairment disease-domain indicator. |
| `other_functional_risk` | binary | Other functional risk-domain indicator. |
| `other` | binary | Unclassified disease-domain indicator, sourced from `unclassified` in the risk-profile table. |

## Lookahead Target Labels

Alerts are first compressed with a 10-minute burst window, retaining only the
first alert in a burst. Labels then represent the presence of a future alert in
the mutually exclusive lookahead interval from `t` exclusive through
`t + 24h` inclusive.

| Column | Type | Definition | Benchmark use |
| --- | --- | --- | --- |
| `label_1` | binary | Low Risk alert within the next 24 hours. | Auxiliary future-risk label. |
| `label_2` | binary | Potential/Unknown alert within the next 24 hours. | Auxiliary future-risk label. |
| `label_3` | binary | Acute Crisis alert within the next 24 hours. | Primary binary benchmark target. |

Although alert severities are mutually exclusive at the alert event level, the
three future-window labels can co-occur when multiple severities occur inside
the same 24-hour lookahead interval. Binary Level 3 benchmarking should treat
`label_3` as the positive class and all windows with `label_3 == 0` as negative,
with optional pure-healthy filtering for anomaly-detector training.
