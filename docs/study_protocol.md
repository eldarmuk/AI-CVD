# Retrospective recorded-alarm prediction protocol — task v2.1.1

`configs/tasks/level3_4h.toml` is authoritative: five-minute grid, 96 rows/eight-hour
lookback, four-hour horizon, primary Level 3. Version 2.1 explicitly changes the
endpoint clock and support policy. Previous results and v2.0 blocked audits are
historical; `docs/archive/study_protocol_v2.0.md` preserves the previous protocol.

The v2.1.1 amendment excludes conflicting valid cumulative snapshots at an identical
timestamp as missing. It never chooses or averages those counters. This fixes a
source ambiguity on which v2.1.0 correctly stopped; endpoint, sampling, intervals
and split are unchanged. The interrupted run records its explicit task amendment
and preserves existing nonconflicting shards byte-for-byte for independent checking.

## Retrospective endpoint

The endpoint is **an alarm initiated at alert_date that was subsequently
documented/classified as a Level-3 emergency escalation**. Later notes supply the
retrospective outcome, never information presumed known at alarm initiation.
The classifier is a keyword proxy, not clinical adjudication. Negatives mean no
qualifying recorded alarm under the completeness assumption, not absence of disease,
unrecorded incidents or unsuccessful alarm attempts.

Adjacent alarms separated by at most ten minutes form chained episodes. Preserve
all alarm IDs/times, episode start/end, first/max/final classification and the first
alarm associated with Level 3. A Level-1 alarm followed by a Level-3-classified alarm
has its primary target time at the latter alarm, not the earlier episode start.
Escalation-decision, dispatch and note-recording times remain NULL when unknown.

Inputs have bucket starts in `[t-480min,t)` and availability no later than t. Targets
use `(t,t+240min]`. Lead time is **minutes to alarm initiation**, never to dispatch
or an escalation decision. An isolated grid-aligned event gives 48 positive times.
All matching episodes are linked by the sample's `[first_event,event_stop)` range
in its patient's ordered target-episode list. Level 2+3 remains a separately
identified secondary task, not the primary compact build.

## Evidence and explicit assumptions

Private metadata distinguishes `researcher_attestation`, `empirical_support`,
`derived_from_source` and `unavailable`. No attestation is labeled clinic-confirmed
or vendor-documented.

- Acquisition time is treated as availability time because the researcher attests
  no expected delayed uploads/backfills. Source timezone is Europe/Warsaw.
- The researcher attests intended completeness of supplied recorded alerts during
  the study period. Continuous monitoring and outages are not independently verified.
- alert_date is researcher-attested button-press time; notes/classification come later.
- Steps are researcher-attested cumulative snapshots, with strong empirical support.
- Enrollment/discharge, outages and clinical-history availability remain unavailable.
  All undated static/history inputs remain NULL with zero known masks.

The exact analysis slice is local `[2025-11-01 00:00,2026-02-01 00:00)`, consistent
with the inspected source extent. Convert using Europe/Warsaw, never naive-as-UTC.
The slice is outside Polish DST transitions; ambiguous/nonexistent times are still
rejected rather than guessed.

## Observed support and censoring — not enrollment

Within the slice, each patient's first and last valid measurement define
`support_start` and support end. They are **not** inferred enrollment/discharge.
A valid raw cumulative counter can establish support even if its increment is
unknown. No-measurement, all-invalid and single-time patients have distinct exclusions.

Construct full buckets from ceil(support_start), closing no later than support end.
The unchanged 24-hour run-in is anchored at support_start as an analysis boundary,
requiring four valid HR buckets and two valid paired-PP buckets. Density uses only
this fixed run-in, not subsequent monitoring. Require 96 contiguous input buckets,
completed run-in, `t-8h >= support_start`, and `t+4h <= support_end`. Apply support
censoring to positives and negatives alike. Generate candidate times independently
of future targets; future-looking labels never select X.

Last-observation censoring is explicitly a **future-monitoring-dependent retrospective
restriction**, not prospective enrollment eligibility. It is not outcome-based
sampling, but could still select for monitoring behavior after events. Internal
outages are not reliably identified. First/last activity does not prove continuous
observability between them: interpretation relies on the researcher-attested
recorded-alert completeness assumption. Report supported-stream prevalence, not
population prevalence, and acknowledge selection/generalizability limitations.

## Features and Steps

Buckets cover `[start,start+5min)` and become available at their end. Invalid values
never reset recency. Masks distinguish missing from observed zero. Shock index is
HR/SBP; PP uses valid paired SBP/DBP. All implementations share configured bounds;
vectorized features are regression-tested against the scalar reference.

Never sum cumulative Steps snapshots. Differences between successive valid,
nondecreasing readings in one Europe/Warsaw local day give observed increments.
Equal counters give observed zero. Initial readings, every cross-day boundary and
unexpected decreases give unknown activity and initialize the next interval.
Invalid observations do not reset counters. Source/reset masks and delta duration
are preserved; no missing activity is fabricated. Deltas are assigned to their
endpoint and may span more than five minutes. Rolling activity sums are partial
observed activity, not exact six-hour totals. Local-day resets are a conservative
processing assumption supported by attestation/patterns, not vendor documentation.

The fixed 59-channel feature order remains schema v2; see the feature dictionary.
Clock features use UTC after conversion. Rolling features use preceding closed
buckets and may have causal warm-up before the first input row. Notes, targets,
episode-wide severity and future monitoring summaries are never input channels.

## Splits, indexed exports, normalization and validation

The unchanged patient hash/seed assigns nominal 70/15/15 partitions before outcomes
or sampling. Streams contain all eligible grid times without class balancing.
Historical test exposure is not erased. Event-free reference training excludes
severity 1/2/3 alarms throughout `[t-8h,t+4h]` in training patients only. This is
explicit retrospective reference selection, not proof of physiological health.

Compact exports store each float32 feature row once per checksummed patient shard,
plus end-row indices, immutable 32-byte sample IDs, targets, lead times, event ranges
and training flags. Split manifests identify task/source checksums.
`compact.sequence_batches` reconstructs exact 96-by-59 arrays with IDs. No sequence
is dropped to reduce storage; overlapping histories are referenced rather than copied.

Normalization fits unique buckets used by event-free training windows only, so
overlapping windows do not reweight a bucket repeatedly. Store observed counts,
population means/scales. Masks, reset/night indicators and sine/cosine keep identity
scaling. All-missing/constant features get scale 1; missing values stay NaN. No
validation/test observations fit statistics.

Independent validation checks hashes, every sample identity, split isolation,
support bounds, alarm-derived targets and lead times, masks/recency/history exclusion,
training selection and normalization recomputed from training rows. No model is
trained. Private reports, hashes and data remain ignored; prior artifacts are preserved.
