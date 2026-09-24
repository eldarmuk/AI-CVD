"""Lossless indexed sequence exports: store each feature row once, preserve sample IDs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd

from .cli import file_hash, localize
from .dataset import patient_split, sample_id
from .episodes import build_episodes, event_time, iso
from .features import FEATURE_NAMES, PHYSIOLOGY, STATIC, validate_source_contract
from .fast_features import read_clean_patient, vector_features
from .task import load_task


def save_json(path, value):
    with open(path,'x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def window_index(frame, episodes, coverage, task):
    """Vectorized equivalent of the canonical label-independent candidate grid."""
    n = len(frame)
    times = frame.index.asi8 + task.grid_minutes*60*10**9
    minute = 60*10**9
    origin = pd.Timestamp(coverage.get('support_start',coverage.get('enrollment_time'))).value
    rin = origin + task.run_in_minutes*minute
    run_in = (frame.index.asi8 >= origin) & (times <= rin)
    density = (frame.loc[run_in,'observed_heartrate'].sum() >= task.min_valid_hr_per_run_in
               and frame.loc[run_in,'observed_pulse_pressure'].sum() >= task.min_valid_bp_per_run_in)
    reason_names = ['insufficient_history','run_in_incomplete','insufficient_run_in_density','outcome_coverage_not_started','insufficient_follow_up']
    flags = [((np.arange(n)+1 < task.sequence_steps) | (times-task.lookback_minutes*minute < origin)),
             times < rin, (times >= rin) & ~np.full(n,density),
             times < pd.Timestamp(coverage['outcome_coverage_start']).value,
             times+task.horizon_minutes*minute > min(pd.Timestamp(coverage['measurement_coverage_end']).value,pd.Timestamp(coverage['outcome_coverage_end']).value)]
    reason = np.full(n,-1,dtype=np.int8)
    for i,flag in enumerate(flags):
        reason[(reason<0)&flag] = i
    candidates = np.arange(n) % task.prediction_stride_steps == 0
    eligible = (reason<0)&candidates
    events = sorted([(pd.Timestamp(event_time(e,task.target_severities)).value,e['episode_id']) for e in episodes if event_time(e,task.target_severities) is not None])
    et = np.array([t for t,_ in events],dtype=np.int64)
    first = np.searchsorted(et,times,side='right')
    stop = np.searchsorted(et,times+task.horizon_minutes*minute,side='right')
    target = first < stop
    lead = np.full(n,np.nan)
    lead[target] = (et[first[target]]-times[target])/minute
    actual = np.array(sorted(pd.Timestamp(m['timestamp']).value for e in episodes for m in e['constituents'] if m['severity'] in task.training_excluded_severities),dtype=np.int64)
    train_ok = np.searchsorted(actual,times+task.horizon_minutes*minute,side='right') == np.searchsorted(actual,times-task.lookback_minutes*minute,side='left')
    train_ok &= times-task.lookback_minutes*minute >= pd.Timestamp(coverage['outcome_coverage_start']).value
    if task.training_policy == 'never_event_patients':
        train_ok &= len(actual)==0
    flow = Counter(candidate_prediction_times=int(candidates.sum()),eligible_prediction_times=int(eligible.sum()),eligible_patients=int(eligible.any()),patients_initially_available=1,events_total=len(events))
    for i,name in enumerate(reason_names):
        flow['excluded_'+name] = int(((reason==i)&candidates).sum())
        flow['patients_excluded_'+name] = int(not eligible.any() and flow['excluded_'+name]>0)
    for at,_ in events:
        opportunities = (times >= at-task.horizon_minutes*minute)&(times<at)&candidates
        found = bool((eligible&opportunities).any())
        flow['events_with_eligible_prediction' if found else 'events_without_eligible_prediction'] += 1
        if not found:
            flow['events_outside_candidate_horizons'] += int(not opportunities.any())
            for flag,name in zip(flags,reason_names):
                flow['events_lost_'+name] += int((opportunities&flag).any())
    ends = np.flatnonzero(eligible).astype(np.int32)
    return {'end_rows':ends,'target':target[eligible].astype(np.uint8),'lead_minutes':lead[eligible].astype(np.float32),
            'training':train_ok[eligible].astype(np.uint8),'first_event':first[eligible].astype(np.int32),
            'event_stop':stop[eligible].astype(np.int32)},dict(flow),events


def sequence_batches(run, split, task, training_only=False, batch_size=256):
    """Verified lazy arrays with IDs; consume the iterator fully to verify the export."""
    run = Path(run)
    meta = json.loads((run/'run_metadata.json').read_text())
    if meta['status'] != 'complete' or meta['task_identifier'] != task.identifier or tuple(meta['feature_names']) != FEATURE_NAMES:
        raise ValueError('Incomplete or incompatible compact dataset')
    verification=json.loads((run/'integrity_verification.json').read_text())
    if verification['status']!='passed' or verification['run_metadata_sha256']!=file_hash(run/'run_metadata.json'):
        raise ValueError('Independent integrity verification is missing or stale')
    export_name = split+('-training' if training_only else '-stream')
    export = json.loads((run/'exports'/f'{export_name}.json').read_text())
    export_digest = file_hash(run/'exports'/f'{export_name}.json')
    if export_digest != meta['artifacts_sha256'][f'exports/{export_name}.json']:
        raise ValueError('Export checksum mismatch')
    if training_only and split != 'train':
        raise ValueError('Training-only normalization/export cannot use held-out patients')
    for shard in export['shards']:
        path = run/shard['file']
        if file_hash(path) != shard['sha256']:
            raise ValueError('Compact shard checksum mismatch')
        with np.load(path,allow_pickle=False) as z:
            X, ends, targets, ids = z['features'], z['end_rows'], z['target'], z['sample_ids']
            selected = np.flatnonzero(z['training']) if training_only else np.arange(len(ends))
            origin = int(z['grid_start_ns'])
            for offset in range(0,len(selected),batch_size):
                sel = selected[offset:offset+batch_size]
                rows = ends[sel,None]-task.sequence_steps+1+np.arange(task.sequence_steps)
                if (rows<0).any() or (ends>=len(X)).any():
                    raise ValueError('Invalid sequence bounds')
                computed = [sample_id(shard['senior_id'],pd.Timestamp(origin+(int(e)+1)*task.grid_minutes*60*10**9,tz='UTC').to_pydatetime()) for e in ends[sel]]
                if any(bytes(ids[j]).hex() != sid.removeprefix('sample_') for j,sid in zip(sel,computed)):
                    raise ValueError('Immutable sample alignment mismatch')
                prediction_times=[iso(pd.Timestamp(origin+(int(e)+1)*task.grid_minutes*60*10**9,tz='UTC').to_pydatetime()) for e in ends[sel]]
                episode_ids=[[event[1] for event in shard['events'][int(z['first_event'][j]):int(z['event_stop'][j])]] for j in sel]
                yield {'X':X[rows], 'y':targets[sel], 'sample_ids':computed,
                       'senior_ids':[shard['senior_id']]*len(sel),'prediction_times':prediction_times,
                       'episode_ids':episode_ids,'lead_minutes':z['lead_minutes'][sel],
                       'split':split,'endpoint':task.endpoint,
                       'sampling':'retrospective_unsupervised_training' if training_only else 'complete_eligible_stream',
                       'task_identifier':task.identifier, 'source_manifest_sha256':export_digest}


def patient_job(raw_db, sid, number, task, contract, output, recovered_coverage=None):
    """One bounded patient transaction, disjoint output file; no shared mutable state."""
    connection=sqlite3.connect(Path(raw_db).resolve().as_uri()+'?mode=ro',uri=True)
    connection.execute('PRAGMA query_only=ON'); connection.execute('BEGIN')
    start=task.study_start_local.replace('T',' '); end=task.study_end_local_exclusive.replace('T',' ')
    try:
        alerts=[dict(zip(('alert_id','senior_id','alert_date','sos_note'),r)) for r in connection.execute(
            'SELECT alert_id,senior_id,alert_date,sos_note FROM alerts WHERE senior_id=? AND alert_date>=? AND alert_date<? ORDER BY alert_date,alert_id',(sid,start,end))]
        for alert in alerts:
            alert['alert_date']=localize(alert['alert_date'],contract)
        episodes=build_episodes(alerts,task)
        existing=Path(output)/f'patients/{number:06d}.npz'
        clean=None if existing.exists() and recovered_coverage else read_clean_patient(connection,sid,task,start,end)
    finally:
        connection.close()
    split=patient_split(sid,task)
    stats=Counter(source_patients=1,level3_episodes_source=sum(e['maximum_severity']==3 for e in episodes),
                  raw_alarm_records=len(alerts),unclassified_alarm_records=sum(m['severity']==-1 for e in episodes for m in e['constituents']))
    result={'episodes':episodes,'split':split,'stats':stats,'flow':{},'coverage':None,'record':None}
    observed=clean[list(PHYSIOLOGY)].notna().any(axis=1) if clean is not None else None
    if clean is not None and (clean.empty or not observed.any()):
        result['flow']={'patients_excluded_no_measurements_in_period' if clean.empty else 'patients_excluded_no_valid_measurements':1}
        return result
    if clean is None:
        first,last=recovered_coverage['support_start'],recovered_coverage['measurement_coverage_end']
    else:
        first,last=clean.loc[observed,'date'].iloc[[0,-1]]
        first,last=localize(first,contract),localize(last,contract)
    if first==last:
        result['flow']={'patients_excluded_single_observation_time':1}; return result
    coverage={'senior_id':sid,'support_start':first,'measurement_coverage_end':last,
              'outcome_coverage_start':first,'outcome_coverage_end':last,
              'coverage_basis':'researcher_attested_complete_alerts_with_observed_support',
              'support_evidence':'derived_from_source','outcome_completeness_evidence':'researcher_attestation',
              'enrollment_discharge':'unavailable'}
    if clean is None:
        with np.load(existing,allow_pickle=False) as prior:
            frame=pd.DataFrame(prior['features'],columns=FEATURE_NAMES,index=pd.date_range(pd.Timestamp(int(prior['grid_start_ns']),tz='UTC'),periods=len(prior['features']),freq=f'{task.grid_minutes}min').as_unit('ns'))
    else:
        frame=vector_features(clean,coverage,task,contract)
    index,flow,events=window_index(frame,episodes,coverage,task)
    stats.update(patients=int(len(index['end_rows'])>0),samples=len(index['end_rows']),positive_samples=int(index['target'].sum()),
                 events_with_eligible_prediction=flow.get('events_with_eligible_prediction',0),events_without_eligible_prediction=flow.get('events_without_eligible_prediction',0))
    result.update(flow=flow,coverage=coverage)
    if not len(index['end_rows']):
        return result
    X=frame.to_numpy(dtype=np.float32)
    times=frame.index[index['end_rows']]+pd.Timedelta(minutes=task.grid_minutes)
    identity=np.array([list(bytes.fromhex(sample_id(sid,t.to_pydatetime()).removeprefix('sample_'))) for t in times],dtype=np.uint8)
    if len(np.unique(identity,axis=0)) != len(identity):
        raise ValueError('Duplicate sample IDs')
    if split!='train':
        index['training'][:]=0
    stats['training_samples']=int(index['training'].sum())
    file=f'patients/{number:06d}.npz'
    if existing.exists():
        with np.load(existing,allow_pickle=False) as prior:
            for key,value in dict(features=X,sample_ids=identity,grid_start_ns=np.array(frame.index[0].value),**index).items():
                if not np.array_equal(prior[key],value,equal_nan=True):
                    raise ValueError('Interrupted shard disagrees with recovered identity/labels: '+key)
    else:
        np.savez_compressed(Path(output)/file,features=X,sample_ids=identity,grid_start_ns=np.array(frame.index[0].value),**index)
    result.update(record={'senior_id':sid,'split':split,'file':file,'sha256':file_hash(Path(output)/file),
                          'samples':len(identity),'training_samples':int(index['training'].sum()),'events':events,'flow':flow},
                  feature_rows=len(X),valid_counts=np.isfinite(X).sum(axis=0),
                  histogram=Counter(str(float(x)) for x in index['lead_minutes'][index['target'].astype(bool)]))
    if split=='train' and index['training'].any():
        diff=np.zeros(len(X)+1,np.int64); ends=index['end_rows'][index['training'].astype(bool)]
        np.add.at(diff,ends-task.sequence_steps+1,1); np.add.at(diff,ends+1,-1)
        values=X[np.cumsum(diff[:-1])>0].astype(np.float64)
        n=np.isfinite(values).sum(axis=0); sums=np.nansum(values,axis=0)
        mean=np.divide(sums,n,out=np.zeros_like(sums),where=n>0)
        result['normalization']=(n,mean,np.nansum((values-mean)**2,axis=0))
    return result


def patient_results(raw_db, ids, task, contract, output, workers, recovered=None):
    iterator=iter(enumerate(ids))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=set()
        def submit():
            item=next(iterator,None)
            if item is not None:
                number,sid=item
                pending.add(pool.submit(patient_job,raw_db,sid,number,task,contract,str(output),(recovered or {}).get(sid)))
        for _ in range(workers*2):
            submit()
        while pending:
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                pending.remove(future)
                yield future.result()
                submit()


def build(raw_db, source_contract, output, task, limit=None, workers=6, resume=False):
    contract = tomllib.loads(Path(source_contract).read_text(encoding='utf-8'))
    validate_source_contract(contract)
    if contract['alert_time_semantics'] != 'alarm_initiation_with_retrospective_classification':
        raise ValueError('Compact retrospective build requires the alarm-initiation endpoint contract')
    if contract.get('evidence',{}).get('outcome_completeness') != 'researcher_attestation':
        raise ValueError('Observed-support build requires explicit researcher-attested outcome completeness')
    if workers<1:
        raise ValueError('workers must be positive')
    def source_state():
        source=Path(raw_db)
        return {p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (source,Path(str(source)+'-wal')) if p.exists()}
    before=source_state()
    provenance=contract.get('provenance',{})
    if before[Path(raw_db).name] != {'bytes':provenance.get('source_db_bytes'),'mtime_ns':provenance.get('source_db_mtime_ns')}:
        raise ValueError('Source metadata differs from the attested fingerprint record')
    output = Path(output)
    recovered={}
    if resume:
        if (output/'run_metadata.json').exists() or any((output/'exports').iterdir()):
            raise ValueError('Recovery only supports interrupted builds before finalization')
        if (output/'source-contract.toml').read_bytes()!=Path(source_contract).read_bytes():
            raise ValueError('Recovery source contract mismatch')
        from .cli import read_jsonl
        recovered={r['senior_id']:r for r in read_jsonl(output/'coverage.jsonl')}
        archive=output/('interruption-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        archive.mkdir()
        save_json(archive/'recovery.json',{'task_identifier':task.identifier,'source_metadata':before,
            'existing_shards_sha256':{p.name:file_hash(p) for p in (output/'patients').glob('*.npz')}})
        for name in ('coverage.jsonl','episodes.jsonl'):
            (output/name).rename(archive/name)
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'patients').mkdir(); (output/'exports').mkdir()
        shutil.copyfile(source_contract,output/'source-contract.toml')
    conn = sqlite3.connect(Path(raw_db).resolve().as_uri()+'?mode=ro',uri=True)
    conn.execute('PRAGMA query_only=ON'); conn.execute('BEGIN')
    ids = [str(r[0]) for r in conn.execute('SELECT id FROM seniors ORDER BY id')]
    if limit is not None:
        ids = ids[:limit]
    start_local = task.study_start_local.replace('T',' ')
    end_local = task.study_end_local_exclusive.replace('T',' ')
    total, split_stats, shards = Counter(), {s:Counter() for s in ('train','validation','test')}, []
    histogram = Counter(); valid_counts = np.zeros(len(FEATURE_NAMES),np.int64); feature_rows=0
    # Welford merge on unique feature rows used by event-free training windows.
    norm_n=np.zeros(len(FEATURE_NAMES),np.int64); norm_mean=np.zeros(len(FEATURE_NAMES)); norm_m2=np.zeros(len(FEATURE_NAMES))
    begun = time.monotonic()
    with open(output/'coverage.jsonl','x',encoding='utf-8') as coverage_file, open(output/'episodes.jsonl','x',encoding='utf-8') as episode_file:
        for number,result in enumerate(patient_results(raw_db,ids,task,contract,output,workers,recovered)):
            split_stats[result['split']].update(result['stats'])
            total.update(result['flow'])
            if result['coverage'] is not None:
                coverage_file.write(json.dumps(result['coverage'])+'\n')
            for episode in result['episodes']:
                episode_file.write(json.dumps(episode)+'\n')
            if result['record'] is not None:
                shards.append(result['record'])
                feature_rows+=result['feature_rows']; valid_counts+=result['valid_counts']
                histogram.update(result['histogram'])
            if 'normalization' in result:
                n,mean,m2=result['normalization']; combined=norm_n+n; delta=mean-norm_mean
                norm_mean+=np.divide(delta*n,combined,out=np.zeros_like(mean),where=combined>0)
                norm_m2+=m2+np.divide(delta**2*norm_n*n,combined,out=np.zeros_like(mean),where=combined>0)
                norm_n=combined
            if number%100==0:
                print(json.dumps({'processed_patients':number+1,'total':len(ids),'eligible_samples':sum(s['samples'] for s in split_stats.values()),'elapsed_seconds':round(time.monotonic()-begun),'free_gb':round(shutil.disk_usage(output).free/1e9,2)}),flush=True)
            if shutil.disk_usage(output).free<5*10**9:
                raise RuntimeError('Stopped before exhausting disk space; run is incomplete')
    shards.sort(key=lambda r:r['file'])
    conn.close()
    if source_state()!=before:
        raise ValueError('Source changed during build; dataset cannot be marked complete')
    for split in split_stats:
        for training in ([False,True] if split=='train' else [False]):
            name=split+('-training' if training else '-stream')
            save_json(output/'exports'/f'{name}.json',{'task_identifier':task.identifier,'feature_names':FEATURE_NAMES,
                     'format':'indexed_sequences_v1','split':split,'training_only':training,
                     'samples':split_stats[split]['training_samples' if training else 'samples'],
                     'shards':[r for r in shards if r['split']==split and (not training or r['training_samples']>0)]})
    scale=np.sqrt(np.divide(norm_m2,norm_n,out=np.zeros_like(norm_mean),where=norm_n>0))
    identity_features=[i for i,f in enumerate(FEATURE_NAMES) if f.startswith(('observed_','known_')) or f in ('is_night','steps_counter_reset','hour_sin','hour_cos')]
    scale[(scale==0)|~np.isfinite(scale)]=1
    norm_mean[identity_features]=0; scale[identity_features]=1
    save_json(output/'normalization.json',{'task_identifier':task.identifier,'feature_names':FEATURE_NAMES,'fit_split':'train',
              'fit_population':'unique buckets used by retrospective event-free training windows; no overlapping-window reweighting',
              'count':norm_n.tolist(),'mean':norm_mean.tolist(),'scale':scale.tolist(),
              'missing_policy':'preserve NaN; downstream imputation must preserve masks','all_missing_features':[FEATURE_NAMES[i] for i in np.flatnonzero(norm_n==0)]})
    save_json(output/'cohort_flow.json',dict(total))
    save_json(output/'statistics.json',{'splits':split_stats,'feature_rows_eligible_patients':feature_rows,
              'feature_nonmissing_bucket_counts':dict(zip(FEATURE_NAMES,valid_counts.tolist())),
              'positive_sample_lead_minutes_histogram':dict(histogram),'lead_definition':'minutes_to_alarm_initiation'})
    artifacts={str(p.relative_to(output)).replace('\\','/'):file_hash(p) for p in output.rglob('*') if p.is_file()}
    save_json(output/'run_metadata.json',{'status':'complete','format':'indexed_sequences_v1','task_identifier':task.identifier,
              'task':task.values,'feature_names':FEATURE_NAMES,'source_contract':contract,'artifacts_sha256':artifacts,
              'normalization_population':'training_only','source_path':str(Path(raw_db).resolve()),'patient_shards':shards,
              'source_fingerprint':provenance,'source_metadata_before_after':before,
              'implementation_sha256':hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),
              'pilot_only':limit is not None,'elapsed_seconds':time.monotonic()-begun})
    print('Completed compact build:',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-db',required=True); p.add_argument('--source-contract',required=True)
    p.add_argument('--output',required=True); p.add_argument('--pilot-patients',type=int); p.add_argument('--workers',type=int,default=6)
    p.add_argument('--resume-interrupted',action='store_true')
    a=p.parse_args(); build(a.raw_db,a.source_contract,a.output,load_task(),a.pilot_patients,a.workers,a.resume_interrupted)


if __name__=='__main__':
    main()
