"""Create a private, manuscript-oriented report only after complete integrity validation."""
import argparse
from collections import Counter
import json
import numpy as np
from pathlib import Path
from .cli import file_hash, read_jsonl
from .features import PHYSIOLOGY
from .dataset import patient_split
from .task import Task


def report(run):
    run=Path(run)
    meta=json.loads((run/'run_metadata.json').read_text())
    stats=json.loads((run/'statistics.json').read_text())
    check=json.loads((run/'integrity_verification.json').read_text())
    flow=json.loads((run/'cohort_flow.json').read_text())
    if check['status']!='passed' or check['run_metadata_sha256']!=file_hash(run/'run_metadata.json'):
        raise ValueError('A verified, unchanged run is required')
    source_confirmation='Synthetic source; no private source fingerprint supplied.'
    if meta['source_fingerprint'].get('source_db_sha256'):
        source_check=json.loads((run/'source_reverification.json').read_text())
        if source_check['status']!='passed' or source_check['sha256']!=meta['source_fingerprint']['source_db_sha256']:
            raise ValueError('Complete source byte fingerprint verification is required')
        source_confirmation='The complete source byte hash was independently recomputed after the build and matched the attested source fingerprint.'
    coverage={r['senior_id'] for r in read_jsonl(run/'coverage.jsonl')}
    supported={r['senior_id'] for r in meta['patient_shards']}
    source_events=Counter(); unsupported=Counter()
    source_patients={s:set() for s in stats['splits']}
    warning_patients={s:set() for s in stats['splits']}
    task=Task(meta['task']).validate()
    for e in read_jsonl(run/'episodes.jsonl'):
        if e['maximum_severity']==3:
            split=patient_split(e['senior_id'],task)
            source_patients[split].add(e['senior_id'])
            if check['eligible_windows_per_episode_by_split'][split].get(e['episode_id'],0)>0:
                warning_patients[split].add(e['senior_id'])
            source_events['total']+=1
            source_events['with_eligible_prediction_patient']+=e['senior_id'] in supported
            unsupported['no_valid_support_patient']+=e['senior_id'] not in coverage
    rows=[]
    for split,s in stats['splits'].items():
        prevalence=f"{100*s.get('positive_samples',0)/s['samples']:.6f}%" if s.get('samples',0) else 'unavailable'
        rows.append(f"| {split} | {s['source_patients']:,} | {s.get('patients',0):,} | {s.get('samples',0):,} | {s.get('positive_samples',0):,} | {s['level3_episodes_source']:,} | {s.get('events_with_eligible_prediction',0):,} | {s.get('training_samples',0):,} | {prevalence} |")
    event_rows=[]
    for split in stats['splits']:
        values=[n for n in check['eligible_windows_per_episode_by_split'][split].values() if n>0]
        distribution=f'{min(values)} / {np.median(values):.1f} / {max(values)} / {np.mean(values):.2f}' if values else 'unavailable'
        event_rows.append(f'| {split} | {len(source_patients[split])} | {len(warning_patients[split])} | {len(values)} | {distribution} |')
    steps=json.loads((run/'steps_verification.json').read_text())
    if steps['status']!='passed' or steps['run_metadata_sha256']!=check['run_metadata_sha256']:
        raise ValueError('Independent source Steps verification is required')
    lead_rows=[]
    for split,lead in check['positive_sample_lead_summary_by_split'].items():
        lead_rows.append('| '+split+' | '+(' | '.join(f"{lead[k]:.2f}" for k in ('min','p25','median','p75','max','mean')) if lead['n'] else 'unavailable | unavailable | unavailable | unavailable | unavailable | unavailable')+' |')
    missing=[]
    for feature in PHYSIOLOGY:
        cells=[]
        for split in ('train','validation','test'):
            denom=check['input_cell_denominator_per_feature_by_split'][split]
            count=check['input_nonmissing_counts_by_split'][split][feature]
            cells.append(f'{100*(1-count/denom):.4f}%' if denom else 'unavailable')
        missing.append('| '+feature+' | '+' | '.join(cells)+' |')
    exclusions='\n'.join(f'| {name} | {value:,} |' for name,value in sorted(flow.items()) if name.startswith(('excluded_','patients_excluded_','events_lost_','events_outside_')))
    text=f'''# Canonical private dataset integrity report — researcher-attested retrospective study

Run: `{run.resolve()}`. Task: `{meta['task_identifier']}`.
Build status: {meta['status']}; integrity validation: {check['status']}.
Pilot only: {meta['pilot_only']}. No model was trained.

## Interpretation and evidence

This is prediction of **alarm initiation followed by retrospective Level-3
classification**. Later sos_note/severity information defines outcomes only; it
never enters input features. This is not prediction of the actual escalation
decision/ambulance-dispatch timestamp, which is unavailable. Keyword classification
is a recorded-emergency proxy, not independent clinical adjudication.

| Declaration | Evidence class |
|---|---|
| Intended completeness of recorded alerts in the supplied study period | researcher_attestation |
| Acquisition time treated as availability; delayed/backfilled uploads not expected | researcher_attestation |
| Europe/Warsaw source timezone and button-press alarm timestamp | researcher_attestation |
| Cumulative Steps snapshots | researcher_attestation plus empirical_support |
| Local-day reset treatment | Explicit conservative transformation; empirically supported, not vendor-documented |
| Patient support boundaries | derived_from_source |
| Enrollment/discharge/outage exceptions | unavailable |
| Severity/note-recording/dispatch timestamps | unavailable; not required for retrospective alarm classification |
| Clinical-history availability | unavailable; all undated static/history inputs excluded |

No declaration is presented as clinic-verified or source-system/vendor documentation.
The prior empirical Steps audit was reused as supporting evidence, not rerun or
promoted to authoritative documentation. Its snapshot identity was not originally
cryptographically bound to the source, a provenance limitation.

## Exact support and censoring rule

Use local `[{meta['task']['study_start_local']},{meta['task']['study_end_local_exclusive']})`
and convert with Europe/Warsaw. For each patient derive the first and last valid
measurement within that slice as analysis support boundaries, not enrollment or
discharge. Start full buckets at ceil(support_start). Require the fixed 24-hour
run-in with four observed HR buckets and two valid paired-PP buckets, 96 contiguous
five-minute rows, `t-8h >= support_start`, and `t+4h <= support_end`. Apply the same
edge censoring to positives and negatives. Targets are alarms in `(t,t+4h]` with
eventual qualifying classification. Density uses only the initial run-in.
The eight hours describe sequence-row timestamps. Precomputed rolling features
and recency/counter state may incorporate earlier causal observations; this is
not a claim that every raw contributing measurement lies within eight hours.

This is explicitly retrospective future-monitoring-dependent support restriction.
Internal outages cannot be certified from first/last readings. Complete capture of
recorded alarms between support boundaries is a researcher-attested assumption;
absence of records does not prove absence of clinical illness or unrecorded alarms.
Post-event monitoring patterns may affect inclusion and generalizability.

## Cohort and natural-prevalence streams

| Split | Source patients | Eligible patients | Stream samples | Positive samples | Source L3 episodes | L3 episodes with eligible prediction | Event-free training samples | Stream prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Total source Level-3 episodes: **{source_events['total']:,}**. Source episodes without
any eligible warning: **{source_events['total']-sum(s.get('events_with_eligible_prediction',0) for s in stats['splits'].values()):,}**.
Of these, episodes in patients without a usable support interval:
**{unsupported['no_valid_support_patient']:,}**. All source episodes remain auditable
in episodes.jsonl, including those outside usable prediction horizons.

Prevalence is the proportion of positive five-minute predictions in the full
supported stream, not patient incidence or prevalence in all monitored elderly
people. Overlapping windows are correlated; confidence intervals must account for
patient/event clustering. No class balancing was applied to the held-out stream.
Patient hashes/seed are unchanged; historical researcher exposure to test data remains.

| Split | Source L3 patients | Patients with an eligible L3 warning | Unique L3 episodes with warning | Windows per warned episode: min / median / max / mean |
|---|---:|---:|---:|---|
{chr(10).join(event_rows)}

An episode is counted once by episode ID, even when it labels many windows. Window
counts per episode include every linked event; a window covering two episodes can
contribute to both event-specific counts, but only once to positive-window prevalence.

Unclassified/unknown-note alarm records by source partition:
`{json.dumps({s:v.get('unclassified_alarm_records',0) for s,v in stats['splits'].items()})}`.
These are not evidence of health; classification sensitivity requires clinical review.
Independent raw-source alarm reconciliation: `{json.dumps(steps['source_alarm_reconciliation'])}`.

## Exclusions and cohort flow

| Criterion | Count |
|---|---:|
{exclusions}

Window exclusions are sequential/disjoint; patient-reason and event-reason counts may overlap.
Events absent from candidate horizons are reported separately. The source/no-support
episode count above closes the gap for patients without a usable measurement interval.
These are analysis exclusions, not proven clinical enrollment/discharge outcomes.

## Lead time to alarm initiation (minutes)

| Split | Minimum | P25 | Median | P75 | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lead_rows)}

This distribution is over positive prediction samples, not one best warning per
episode. Every lead time was independently checked against the first qualifying
future alarm. It is not lead time to escalation decision or dispatch.

## Missingness in actual input windows

| Channel | Train missing | Validation missing | Test missing |
|---|---:|---:|---:|
{chr(10).join(missing)}

Denominators count input cells across all eligible 96-row windows, including repeated
buckets in overlapping histories. Full 59-channel observed counts and denominators
are in integrity_verification.json; unique patient-grid summaries are in statistics.json.
Observed zero activity remains distinct from missing activity. Physiological invalid
values become missing and cannot reset recency. Static/history channels are all
unknown, with known masks zero.

## Steps and normalization

Within one Europe/Warsaw local day, differences of successive valid nondecreasing
cumulative snapshots give increments. Equal values give observed zero. Initial
readings, every local-day boundary and unexpected decreases give unknown activity
and restart differencing. Invalid readings do not reset the counter. Raw snapshots
are never summed. Source/reset masks and delta duration are retained; long deltas
and partial rolling sums are not exact five-minute/six-hour activity totals.

Independent comparison against raw cumulative snapshots passed for all eligible
patient shards. Audit counts (snapshots/increments/buckets are distinct units):
`{json.dumps(steps['totals'])}`.

Normalization was fitted only on unique feature buckets referenced by retrospective
event-free training windows. No validation/test values were fitted. Means, population
scales, observed counts and all-missing channels are in normalization.json; masks
and clock encodings retain identity scaling. Missing values remain NaN. The verifier
independently recomputed statistics from the training selection.

## Reproducibility and verification

Features are stored once per patient; indexed exports reconstruct each exact
96-by-59 sequence with its immutable sample ID, prediction time, target, episode
range and lead time. Stream train/validation/test exports and event-free train
exports live under exports/. This is lossless sequence indexing, not subsampling.

- Task fingerprint: `{meta['task_identifier']}`
- Source fingerprint/provenance: `{json.dumps(meta['source_fingerprint'])}`
- {source_confirmation}
- Implementation SHA-256: `{meta['implementation_sha256']}`
- Run metadata SHA-256: `{check['run_metadata_sha256']}`
- Every build artifact is checksummed in run_metadata.json. Integrity verification
  independently checked all IDs, splits, support bounds, targets, lead times,
  masks/recencies, excluded history and training-only normalization.

## Readiness

Under the explicitly researcher-attested retrospective design, this verified run
is suitable for proceeding to model development only if `pilot_only` is false.
It does not establish prospective performance, continuous clinical surveillance,
independent endpoint adjudication or a newly untouched test population. Keep these
limitations in the manuscript. No CCA-TAVAE training has been executed.
'''
    with open(run/'dataset_integrity_report.md','x',encoding='utf-8') as f:
        f.write(text)
    compact=f'''# Manuscript cohort summary — retrospective recorded-alarm prediction

Task: {meta['task_identifier']}. Five-minute grid; 96-row/eight-hour input; four-hour
horizon; retrospective Level-3 classification at alarm initiation. Researcher-attested
availability/completeness/timezone; undated clinical histories excluded.

| Split | Source patients | Eligible patients | Stream windows | Positive windows | Source L3 episodes | L3 episodes with warning | Event-free training windows | Window prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

| Split | Source L3 patients | Patients with an eligible L3 warning | Unique L3 episodes with warning | Windows per warned episode: min / median / max / mean |
|---|---:|---:|---:|---|
{chr(10).join(event_rows)}

| Split | Lead min | P25 | Median | P75 | Max | Mean (minutes) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lead_rows)}

Leads are to alarm initiation, summarized over correlated positive windows.
Clinical episodes are unique episode IDs, never the number of positive windows.
Require completed fixed run-in and t+4h within last-valid-observation support.
Last activity is not discharge: this retrospective restriction may introduce
selection bias when monitoring ceases after an event. Internal outages and continuous
outcome observability remain unverified. Recorded-alert completeness is assumed.
Keyword outcomes need independent clinical review; historic test exposure remains.
Full exclusions, missingness, provenance and Steps checks: dataset_integrity_report.md.
'''
    with open(run/'manuscript_cohort_summary.md','x',encoding='utf-8') as f:
        f.write(compact)
    return run/'dataset_integrity_report.md'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    print(report(p.parse_args().run))


if __name__=='__main__':
    main()
