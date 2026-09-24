"""Independently compare private cumulative-counter features to source snapshots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from .cli import file_hash, localize, read_jsonl
from .episodes import build_episodes
from .features import FEATURE_NAMES
from .task import load_task


def audit(run, raw_db, task):
    run=Path(run);meta=json.loads((run/'run_metadata.json').read_text())
    contract=meta['source_contract']
    if contract['steps_semantics']!='cumulative_counter':
        raise ValueError('This audit is for cumulative snapshots only')
    source=Path(raw_db);before=(source.stat().st_size,source.stat().st_mtime_ns)
    assert before==(contract['provenance']['source_db_bytes'],contract['provenance']['source_db_mtime_ns'])
    coverage={r['senior_id']:r for r in map(json.loads,(run/'coverage.jsonl').read_text().splitlines())}
    c=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
    c.execute('PRAGMA query_only=ON')
    patients={str(r[0]) for r in c.execute('SELECT id FROM seniors')}
    alerts=[];orphans=0
    for row in c.execute('SELECT alert_id,senior_id,alert_date,sos_note FROM alerts WHERE alert_date>=? AND alert_date<? ORDER BY senior_id,alert_date,alert_id',(task.study_start_local.replace('T',' '),task.study_end_local_exclusive.replace('T',' '))):
        if str(row[1]) not in patients:
            orphans+=1
            continue
        a=dict(zip(('alert_id','senior_id','alert_date','sos_note'),row))
        a['senior_id']=str(a['senior_id']);a['alert_date']=localize(a['alert_date'],contract);alerts.append(a)
    expected=sorted(build_episodes(alerts,task),key=lambda e:e['episode_id'])
    actual=sorted(read_jsonl(run/'episodes.jsonl'),key=lambda e:e['episode_id'])
    assert expected==actual,'Episodes do not reconcile with source alarms'
    statistics=json.loads((run/'statistics.json').read_text())
    assert sum(s['source_patients'] for s in statistics['splits'].values())==len(patients)
    assert sum(s['raw_alarm_records'] for s in statistics['splits'].values())==len(alerts)
    totals=Counter();by_split={s:Counter() for s in ('train','validation','test')}
    for number,shard in enumerate(meta['patient_shards']):
        sid=shard['senior_id'];support=coverage[sid]
        raw=pd.read_sql_query("SELECT date,value FROM measurements WHERE senior_id=? AND type='Steps' AND date>=? AND date<? ORDER BY date",c,params=(sid,task.study_start_local.replace('T',' '),task.study_end_local_exclusive.replace('T',' ')))
        raw['value']=pd.to_numeric(raw.value,errors='coerce')
        stats=Counter(patients=1,raw_snapshots=len(raw),invalid_snapshots=int((~raw.value.between(*task.validity['steps'])).sum()))
        valid=raw[raw.value.between(*task.validity['steps'])].copy()
        conflicts=valid.groupby('date').value.nunique()>1
        stats['conflicting_timestamps']=int(conflicts.sum())
        ambiguous=valid.date.isin(conflicts[conflicts].index)
        stats['conflicting_snapshot_records_excluded']=int(ambiguous.sum())
        if ambiguous.any() and task.values.get('steps_duplicate_conflict_policy','reject')=='reject':
            raise ValueError('Conflicting source Steps snapshot')
        valid=valid[~ambiguous]
        stats['duplicate_valid_snapshots']=int(valid.duplicated('date').sum())
        valid=valid.drop_duplicates('date')
        at=pd.to_datetime(valid.date,format='mixed').dt.tz_localize(contract['source_timezone'],ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')
        valid.index=pd.DatetimeIndex(at)
        valid=valid[(valid.index>=pd.Timestamp(support['support_start']))&(valid.index<pd.Timestamp(support['measurement_coverage_end']))]
        local_day=pd.Series(valid.index.tz_convert(contract['source_timezone']).date,index=valid.index)
        delta=valid.value.diff();day_change=local_day.ne(local_day.shift())&delta.notna()
        decrease=delta.lt(0);reset=day_change|decrease
        increments=delta.where(~reset & delta.between(*task.validity['steps']))
        interval=pd.Series(valid.index,index=valid.index).diff().dt.total_seconds()/60
        interval=interval.where(increments.notna())
        stats.update(valid_supported_snapshots=len(valid),day_boundaries=int(day_change.sum()),within_day_decreases=int((decrease&~day_change).sum()),cross_day_decreases=int((decrease&day_change).sum()),zero_increments=int(increments.eq(0).sum()),positive_increments=int(increments.gt(0).sum()),increments_longer_than_grid=int(interval.gt(task.grid_minutes).sum()))
        with np.load(run/shard['file'],allow_pickle=False) as z:
            grid=pd.date_range(pd.Timestamp(int(z['grid_start_ns']),tz='UTC'),periods=len(z['features']),freq=f'{task.grid_minutes}min')
            buckets=valid.index.floor(f'{task.grid_minutes}min')
            expected={
                'steps':increments.groupby(buckets).sum(min_count=1).reindex(grid),
                'steps_source_value':valid.value.groupby(buckets).last().reindex(grid),
                'steps_counter_reset':reset.groupby(buckets).max().reindex(grid).fillna(False).astype(float),
                'steps_delta_interval_minutes':interval.groupby(buckets).max().reindex(grid)}
            expected['observed_steps']=expected['steps'].notna().astype(float)
            expected['observed_steps_source']=expected['steps_source_value'].notna().astype(float)
            for name,values in expected.items():
                np.testing.assert_allclose(z['features'][:,FEATURE_NAMES.index(name)],values.to_numpy(),rtol=1e-6,atol=1e-5,equal_nan=True,err_msg=name)
            stats['feature_buckets']=len(grid)
            stats['missing_increment_buckets']=int(expected['steps'].isna().sum())
            stats['observed_zero_buckets']=int(expected['steps'].eq(0).sum())
        totals.update(stats);by_split[shard['split']].update(stats)
        if number%500==0:
            print(json.dumps({'steps_verified_patients':number+1}),flush=True)
    c.close()
    assert before==(source.stat().st_size,source.stat().st_mtime_ns)
    result={'status':'passed','scope':'all eligible patient shards; raw snapshots within study slice, increments within analysis support',
            'totals':totals,'by_split':by_split,'source_alarm_reconciliation':{'matched_records':len(alerts),'matched_episodes':len(actual),'orphan_alarm_records_without_patient_table_entry':orphans},
            'run_metadata_sha256':file_hash(run/'run_metadata.json')}
    with open(run/'steps_verification.json','x') as f:
        json.dump(result,f,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--raw-db',required=True)
    a=p.parse_args();print(json.dumps(audit(a.run,a.raw_db,load_task()),indent=2))


if __name__=='__main__':
    main()
