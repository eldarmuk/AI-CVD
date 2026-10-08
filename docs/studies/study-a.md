# Study A — repaired continuous early warning

**Status: internally completed and frozen; not independently verified as published.**

Study A defines prediction times before a future alarm, with eight hours of
five-minute wearable history and a four-hour forecasting horizon. The target is
recorded alarm initiation subsequently classified Level 3, not a prospectively
adjudicated cardiovascular event or the time severity became known.

Repairs separated the primary task from earlier exploratory analyses and clarified
causal histories, valid observations, subject-level partitioning and evaluation.
R0/R1 observation-aware encoder definitions are retained in
[continuous.py](../../src/ai_cvd/models/continuous.py). The
[public tensor adapter](../../src/ai_cvd/features/continuous.py) accepts normalized
in-memory tensors; it does not reproduce the private source-schema projection.

The public tests exercise model shapes, artificial masking, reconstruction loss and
frozen-head interfaces. Private preprocessing, fitted encoders, thresholds, predictions
and exact run bindings are not included. This candidate does not publish additional
Study A numerical results without a separately approved aggregate record.

Historical bugs, superseded results and unsuccessful experiments remain in the
preserved private record. A legacy architecture appearing in the package does not
make its earlier performance directly comparable to this repaired task.
