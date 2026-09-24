"""End-to-end integrity checks for compact private runs; never fits or trains a model."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .cli import file_hash, read_jsonl
from .dataset import patient_split, sample_id
from .features import FEATURE_NAMES, PHYSIOLOGY, STATIC
from .episodes import event_time
from .task import load_task


def validate(run, task):
    run=Path(run); meta=json.loads((run/'run_metadata.json').read_text())
    if meta['status']!='complete' or meta['task_identifier']!=task.identifier or tuple(meta['feature_names'])!=FEATURE_NAMES:
        raise ValueError('Task/schema/completion mismatch')
    for name,digest in meta['artifacts_sha256'].items():
        if not (run/name).resolve().is_relative_to(run.resolve()) or file_hash(run/name)!=digest:
            raise ValueError('Artifact checksum mismatch: '+name)
    for journal in run.glob('interruption-*/recovery.json'):
        recovery=json.loads(journal.read_text())
        if recovery['task_identifier']!=task.identifier:
            from .task import Task
            amendment=json.loads((run/'task_amendment.json').read_text())
            old,new=amendment['from_task'],amendment['to_task']
            assert Task(old).identifier==recovery['task_identifier'] and new==task.values
            assert {k for k in set(old)|set(new) if old.get(k)!=new.get(k)}=={'version','steps_duplicate_conflict_policy'}
            assert old.get('steps_duplicate_conflict_policy','reject')=='reject' and new['steps_duplicate_conflict_policy']=='exclude_timestamp'
        for name,digest in recovery['existing_shards_sha256'].items():
            assert meta['artifacts_sha256']['patients/'+name]==digest,'Recovered shard was modified'
    coverage={r['senior_id']:r for r in read_jsonl(run/'coverage.jsonl')}
    episodes={}
    for episode in read_jsonl(run/'episodes.jsonl'):
        episodes.setdefault(episode['senior_id'],[]).append(episode)
        assert episode['maximum_severity']==max(r['severity'] for r in episode['constituents'])
        assert episode['escalation_recorded_at'] is None
    seen=set(); counts={s:Counter() for s in ('train','validation','test')}
    input_valid={s:np.zeros(len(FEATURE_NAMES),np.int64) for s in counts}
    positive_leads={s:[] for s in counts}
    episode_windows={s:{} for s in counts}
    observed_window_hist={s:{f:np.zeros(task.sequence_steps+1,np.int64) for f in list(PHYSIOLOGY)+['any_physiology']} for s in counts}
    minute=60*10**9; names=list(FEATURE_NAMES)
    norm_n=np.zeros(len(names),np.int64); norm_sum=np.zeros(len(names)); norm_ss=np.zeros(len(names))
    begun=time.monotonic()
    for number,shard in enumerate(meta['patient_shards']):
        sid,split=shard['senior_id'],shard['split']
        assert meta['artifacts_sha256'][shard['file']]==shard['sha256']
        assert sid not in seen and split==patient_split(sid,task)
        seen.add(sid); c=coverage[sid]
        with np.load(run/shard['file'],allow_pickle=False) as z:
            X,ends,ids,y,lead=z['features'],z['end_rows'],z['sample_ids'],z['target'],z['lead_minutes']
            origin=int(z['grid_start_ns']); t=origin+(ends.astype(np.int64)+1)*task.grid_minutes*minute
            assert origin==pd.Timestamp(c['support_start']).ceil(f'{task.grid_minutes}min').value
            assert X.shape[1]==len(names) and ids.shape==(len(ends),32)
            assert np.all(np.diff(ends)>0) and np.all(ends>=task.sequence_steps-1) and np.all(ends<len(X))
            assert np.all(t-task.lookback_minutes*minute>=pd.Timestamp(c['support_start']).value)
            assert np.all(t+task.horizon_minutes*minute<=pd.Timestamp(c['outcome_coverage_end']).value)
            assert np.all(t+task.horizon_minutes*minute<=pd.Timestamp(c['measurement_coverage_end']).value)
            assert np.all(t>=pd.Timestamp(c['outcome_coverage_start']).value)
            assert np.all(t>=pd.Timestamp(c['support_start']).value+task.run_in_minutes*minute)
            bucket_start=origin+np.arange(len(X))*task.grid_minutes*minute
            run_in=(bucket_start>=pd.Timestamp(c['support_start']).value)&(bucket_start+task.grid_minutes*minute<=pd.Timestamp(c['support_start']).value+task.run_in_minutes*minute)
            assert X[run_in,names.index('observed_heartrate')].sum()>=task.min_valid_hr_per_run_in
            assert X[run_in,names.index('observed_pulse_pressure')].sum()>=task.min_valid_bp_per_run_in
            end_ns=pd.Timestamp(c['measurement_coverage_end']).floor(f'{task.grid_minutes}min').value
            assert len(X)==(end_ns-origin)//(task.grid_minutes*minute)
            all_t=bucket_start+task.grid_minutes*minute
            complete=(np.arange(len(X))>=task.sequence_steps-1)&(np.arange(len(X))%task.prediction_stride_steps==0)
            complete&=all_t-task.lookback_minutes*minute>=pd.Timestamp(c['support_start']).value
            complete&=all_t>=pd.Timestamp(c['support_start']).value+task.run_in_minutes*minute
            complete&=all_t>=pd.Timestamp(c['outcome_coverage_start']).value
            complete&=all_t+task.horizon_minutes*minute<=min(pd.Timestamp(c['measurement_coverage_end']).value,pd.Timestamp(c['outcome_coverage_end']).value)
            assert np.array_equal(ends,np.flatnonzero(complete)),'Stream is not the complete eligible grid'
            for j in sorted(set([0,len(ends)//2,len(ends)-1])):
                rows=np.arange(ends[j]-task.sequence_steps+1,ends[j]+1)
                assert len(rows)==96 and bucket_start[rows[0]]==t[j]-task.lookback_minutes*minute
                assert np.all(bucket_start[rows]<t[j])
            # Independent canonical JSON encoding; avoid constructing a datetime
            # object and a 32-element Python list for every stream window.
            assert np.all(t%(10**9)==0)
            stamps=t.view('datetime64[ns]').astype('datetime64[s]').astype(str)
            prefix='['+json.dumps(str(sid))+',"'
            expected=np.frombuffer(b''.join(hashlib.sha256((prefix+stamp+'Z"]').encode()).digest() for stamp in stamps),dtype=np.uint8).reshape(-1,32)
            for j in sorted(set([0,len(t)//2,len(t)-1])):
                assert bytes(expected[j]).hex()==sample_id(sid,pd.Timestamp(int(t[j]),tz='UTC').to_pydatetime()).removeprefix('sample_')
            assert np.array_equal(ids,expected),'Sample identity/order mismatch'
            events=sorted(pd.Timestamp(event_time(e,task.target_severities)).value for e in episodes.get(sid,[]) if event_time(e,task.target_severities) is not None)
            linked=sorted((pd.Timestamp(event_time(e,task.target_severities)).value,e['episode_id']) for e in episodes.get(sid,[]) if event_time(e,task.target_severities) is not None)
            assert [list(e) for e in linked]==shard['events'],'Episode manifest mismatch'
            assert np.array_equal(z['first_event'],np.searchsorted(events,t,side='right'))
            assert np.array_equal(z['event_stop'],np.searchsorted(events,t+task.horizon_minutes*minute,side='right'))
            for at,eid in linked:
                episode_windows[split][eid]=int(((t<at)&(at<=t+task.horizon_minutes*minute)).sum())
            expected_y=np.zeros(len(t),bool); expected_lead=np.full(len(t),np.nan)
            for at in reversed(events):
                positive=(t<at)&(at<=t+task.horizon_minutes*minute)
                expected_y|=positive; expected_lead[positive]=(at-t[positive])/minute
            assert np.array_equal(y,expected_y)
            np.testing.assert_allclose(lead,expected_lead,atol=2e-5,rtol=1e-6,equal_nan=True)
            assert not np.isinf(X).any()
            for f in PHYSIOLOGY:
                assert np.array_equal(np.isfinite(X[:,names.index(f)]),X[:,names.index('observed_'+f)]==1)
                recency=X[:,names.index('time_since_last_'+f)]
                assert np.all(recency[np.isfinite(recency)]>=0)
            for f in STATIC:
                assert np.isnan(X[:,names.index(f)]).all() and (X[:,names.index('known_'+f)]==0).all()
            training=z['training'].astype(bool)
            expected_training=t-task.lookback_minutes*minute>=pd.Timestamp(c['outcome_coverage_start']).value
            actual_alarms=[pd.Timestamp(m['timestamp']).value for e in episodes.get(sid,[]) for m in e['constituents'] if m['severity'] in task.training_excluded_severities]
            for at in actual_alarms:
                expected_training&=~((t-task.lookback_minutes*minute<=at)&(at<=t+task.horizon_minutes*minute))
            if split!='train' or (task.training_policy=='never_event_patients' and actual_alarms):
                expected_training[:]=False
            assert np.array_equal(training,expected_training),'Training selection differs from declared policy'
            window_weights=np.zeros(len(X)+1,np.int64)
            np.add.at(window_weights,ends-task.sequence_steps+1,1);np.add.at(window_weights,ends+1,-1)
            window_weights=np.cumsum(window_weights[:-1])
            input_valid[split]+=np.sum(np.isfinite(X)*window_weights[:,None],axis=0)
            observed=np.isfinite(X[:,[names.index(f) for f in PHYSIOLOGY]])
            for k,f in enumerate(list(PHYSIOLOGY)+['any_physiology']):
                available=observed[:,k] if k<len(PHYSIOLOGY) else observed.any(axis=1)
                prefix=np.r_[0,np.cumsum(available)]
                observed_counts=prefix[ends+1]-prefix[ends-task.sequence_steps+1]
                observed_window_hist[split][f]+=np.bincount(observed_counts,minlength=task.sequence_steps+1)
            positive_leads[split].extend(lead[y.astype(bool)].tolist())
            if split!='train':
                assert not training.any()
            elif training.any():
                for e in episodes.get(sid,[]):
                    for member in e['constituents']:
                        if member['severity'] in task.training_excluded_severities:
                            at=pd.Timestamp(member['timestamp']).value
                            assert not ((t[training]-task.lookback_minutes*minute<=at)&(at<=t[training]+task.horizon_minutes*minute)).any()
                diff=np.zeros(len(X)+1,np.int64)
                np.add.at(diff,ends[training]-task.sequence_steps+1,1);np.add.at(diff,ends[training]+1,-1)
                values=X[np.cumsum(diff[:-1])>0].astype(float)
                norm_n+=np.isfinite(values).sum(axis=0);norm_sum+=np.nansum(values,axis=0);norm_ss+=np.nansum(values**2,axis=0)
            counts[split].update(patients=1,samples=len(ends),positive_samples=int(y.sum()),training_samples=int(training.sum()))
        if number%200==0:
            print(json.dumps({'verified_patients':number+1,'elapsed_seconds':round(time.monotonic()-begun)}),flush=True)
    stats=json.loads((run/'statistics.json').read_text()); normal=json.loads((run/'normalization.json').read_text())
    assert normal['fit_split']=='train' and normal['task_identifier']==task.identifier
    assert np.array_equal(norm_n,np.array(normal['count']))
    mean=np.divide(norm_sum,norm_n,out=np.zeros_like(norm_sum),where=norm_n>0)
    scale=np.sqrt(np.maximum(0,np.divide(norm_ss,norm_n,out=np.zeros_like(norm_ss),where=norm_n>0)-mean**2));scale[scale==0]=1
    for i,f in enumerate(names):
        if f.startswith(('observed_','known_')) or f in ('is_night','steps_counter_reset','hour_sin','hour_cos'):
            mean[i]=0;scale[i]=1
    np.testing.assert_allclose(mean,normal['mean'],rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(scale,normal['scale'],rtol=1e-4,atol=1e-4)
    for split in counts:
        for key,value in counts[split].items():
            assert value==stats['splits'][split][key]
        export=json.loads((run/'exports'/f'{split}-stream.json').read_text())
        assert export['task_identifier']==task.identifier and tuple(export['feature_names'])==FEATURE_NAMES
        assert export['samples']==counts[split]['samples']
        assert export['shards']==[r for r in meta['patient_shards'] if r['split']==split]
    training_export=json.loads((run/'exports/train-training.json').read_text())
    assert training_export['shards']==[r for r in meta['patient_shards'] if r['split']=='train' and r['training_samples']>0]
    assert training_export['samples']==counts['train']['training_samples']
    result={'status':'passed','task_identifier':task.identifier,'counts':counts,
            'checks':['artifact_checksums','patient_isolation','all_sample_ids','lookback_and_censoring','retrospective_alarm_targets',
                      'all_lead_times','mask_and_recency','history_exclusion','training_selection','training_only_normalization','episode_linkage','export_alignment','complete_eligible_stream'],
            'eligible_windows_per_episode_by_split':episode_windows,
            'observed_bucket_count_per_window_histogram':{s:{f:h.tolist() for f,h in fs.items()} for s,fs in observed_window_hist.items()},
            'input_nonmissing_counts_by_split':{s:dict(zip(names,v.tolist())) for s,v in input_valid.items()},
            'input_cell_denominator_per_feature_by_split':{s:counts[s]['samples']*task.sequence_steps for s in counts},
            'positive_sample_lead_summary_by_split':{s:({'n':len(v),'min':min(v),'p25':float(np.quantile(v,.25)),'median':float(np.median(v)),
                                                        'p75':float(np.quantile(v,.75)),'max':max(v),'mean':float(np.mean(v))} if v else {'n':0,'distribution':'unavailable_no_positive_samples'}) for s,v in positive_leads.items()},
            'run_metadata_sha256':file_hash(run/'run_metadata.json'),'elapsed_seconds':time.monotonic()-begun}
    with open(run/'integrity_verification.json','x') as f:
        json.dump(result,f,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    a=p.parse_args();print(json.dumps(validate(a.run,load_task()),indent=2))


if __name__=='__main__':
    main()
