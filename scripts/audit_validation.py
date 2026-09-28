"""MANUAL full validation audit/report/freeze. No training and no test access."""
import argparse
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scripts.validation_support import metadata,read,MODEL,SCORES,POLICY
from scripts.validation_audit_math import fast_alerts,accumulator,add,summary,stream_metrics
from src.architecture_study.config import ROOT,fingerprint
from src.architecture_study.data import digest,write_json
from src.ai_cvd.dataset import sample_id
from src.ai_cvd.features import FEATURE_NAMES
from src.architecture_study.projection import PRIMITIVES
from src.final_study.policy import calibrated,calibration_partition
from src.final_study.config import seal


def close(actual,expected):
    if isinstance(expected,dict):
        for k,v in expected.items(): close(actual[k],v)
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise ValueError('List length mismatch')
        for a,b in zip(actual,expected): close(a,b)
    elif isinstance(expected,float): np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
    elif actual!=expected: raise ValueError('Audit value mismatch')


def audit(output):
    if output.exists() or not output.resolve().is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Use a new immutable ignored output directory; partial audits are preserved')
    begun=time.perf_counter()
    spec,c,task,model,scaler,root,manifest,export,cal,pol=metadata()
    output.mkdir(parents=True)
    write_json(output/'audit_protocol.json',{'scope':'canonical validation only, no fitting',
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'scripts').glob('*validation*.py'))},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'spec_sha256':fingerprint(spec),'resume':'unsupported; preserve partial audit'})
    count=sum(r['samples'] for r in manifest)
    arrays={k:np.lib.format.open_memmap(output/f'{k}.npy',mode='w+',dtype=dt,shape=(count,))
            for k,dt in [('scores','float64'),('labels','uint8'),('vital','uint8'),('primitive','uint8'),('role','uint8')]}
    fitted=read(POLICY/'calibration.json'); threshold=read(POLICY/'threshold.json'); recorded=read(POLICY/'policy_development_metrics.json')
    b=fitted['intercept']; cutoff=threshold['selected']['threshold']; cooldown=spec['validation']['cooldown_minutes']
    candidate_acc=[accumulator() for _ in threshold['candidates']]
    all_acc=accumulator(); role_acc={0:accumulator(),1:accumulator()}; position=0
    cal_sum=cal_y=cal_n=0; patients_by_stratum={True:0,False:0}; unique_episodes=set(); event_patients=set()
    for ordinal,(r,s) in enumerate(zip(manifest,export['shards'])):
        source=(root/s['file']).resolve(); pred=(SCORES/r['file']).resolve()
        if not source.is_relative_to((root/'patients').resolve()) or not pred.is_relative_to(SCORES.resolve()): raise ValueError('Invalid patient path')
        if digest(source)!=s['sha256'] or digest(pred)!=r['sha256']: raise ValueError('Patient artifact hash mismatch')
        with np.load(source,allow_pickle=False) as z:
            ends=z['end_rows'].astype(np.int64); start=int(z['grid_start_ns']); ids=z['sample_ids']; labels=z['target']
            features=z['features']
        with np.load(pred,allow_pickle=False) as z:
            d={k:z[k] for k in ('sample_ids','times','scores','vital','labels')}
        n=r['samples']; times=start+(ends+1)*task.grid_minutes*60*10**9
        if n!=len(ends) or any(len(v)!=n for v in d.values()) or np.any(np.diff(ends)<=0): raise ValueError('Row/order mismatch')
        expected_ids=['sample_'+bytes(v).hex() for v in ids]
        if d['sample_ids'].tolist()!=expected_ids or len(set(expected_ids))!=n: raise ValueError('Sample identity mismatch')
        # Recompute immutable IDs directly from patient and acquisition-grid cutoff.
        for sid,t in zip(expected_ids,times):
            if sid!=sample_id(r['patient'],datetime.fromtimestamp(int(t)/1e9,timezone.utc)): raise ValueError('Sample ID/time mismatch')
        np.testing.assert_array_equal(d['times'],times); np.testing.assert_array_equal(d['labels'],labels)
        if not np.isfinite(d['scores']).all() or np.any((d['scores']<0)|(d['scores']>1)): raise ValueError('Invalid risk scores')
        natural=np.isfinite(features[:,[FEATURE_NAMES.index(p) for p in PRIMITIVES]])
        evidence=[]
        for cols in (5,6):
            prefix=np.r_[0,np.cumsum(natural[:,:cols].any(1))]
            evidence.append(prefix[ends+1]-prefix[ends-task.sequence_steps+1]>0)
        np.testing.assert_array_equal(d['vital'],evidence[0])
        event_times=np.array([e[0] for e in s['events']],dtype=np.int64)
        expected_y=np.searchsorted(event_times,times+task.horizon_minutes*60*10**9,side='right')>np.searchsorted(event_times,times,side='right')
        np.testing.assert_array_equal(labels,expected_y)
        role=int(not calibration_partition(r['patient'],spec['validation']['seed']))
        for key,value in [('scores',d['scores']),('labels',labels),('vital',evidence[0]),('primitive',evidence[1]),('role',role)]:
            arrays[key][position:position+n]=value
        position+=n
        pc=calibrated(d['scores'],b)
        result=fast_alerts(times,pc,s['events'],cutoff,task,cooldown,evidence[0])
        add(all_acc,result); add(role_acc[role],result)
        unique_episodes.update((r['patient'],e) for e in result['eligible_episodes'])
        if result['eligible_episodes']: event_patients.add(r['patient'])
        for v in (True,False): patients_by_stratum[v]+=int(np.any(evidence[0]==v))
        if role==0:
            cal_sum+=float(pc.sum()); cal_y+=int(labels.sum()); cal_n+=n
        else:
            close(result,recorded['patients'][r['patient']])
            for a,row in zip(candidate_acc,threshold['candidates']):
                add(a,fast_alerts(times,pc,s['events'],row['threshold'],task,cooldown,evidence[0]))
        print(f'Audited validation patient {ordinal+1}/{len(manifest)}',flush=True)
    if position!=count: raise ValueError('Missing stream rows')
    for a in arrays.values(): a.flush()
    if fitted['samples']!=cal_n or fitted['positives']!=cal_y: raise ValueError('Calibration population differs')
    if fitted['status']=='fitted':
        if not 0<cal_y<cal_n or abs(cal_sum-cal_y)>1e-6: raise ValueError('Intercept is not validation-only likelihood solution')
    elif fitted['status']!='identity_insufficient_classes' or 0<cal_y<cal_n or b!=0:
        raise ValueError('Invalid calibration fallback')
    quantiles=np.unique(np.r_[np.linspace(0,.9,10),1-np.geomspace(.1,1e-6,101),1.])
    selected=arrays['role']==1
    expected_candidates=np.unique(np.r_[np.quantile(calibrated(arrays['scores'][selected],b),quantiles),1.000000000001])
    np.testing.assert_array_equal(expected_candidates,[r['threshold'] for r in threshold['candidates']])
    keys=('eligible_episodes','detected_episodes','episode_sensitivity','alerts','alert_precision','false_alerts','supported_patient_days',
          'false_alerts_per_supported_day','supported_cells','supported_fraction_of_within_patient_span','lead_minutes_quantiles')
    for a,row in zip(candidate_acc,threshold['candidates']):
        sm=summary(a,task)
        close({k:sm[k] for k in keys},{k:row[k] for k in keys})
    # Selected per-patient records and every candidate's operating metrics have now replayed.
    report={'status':'passed','scope':'complete canonical validation audit; no test access',
        'scored_windows':count,'patients':len(manifest),'eligible_unique_episodes':len(unique_episodes),'event_patients':len(event_patients),
        'calibration_parameter':fitted,'selected_threshold':cutoff,'discrimination':{},
        'policy_partition_operational':summary(role_acc[1],task),'calibration_partition_operational_descriptive':summary(role_acc[0],task),
        'whole_validation_operational_descriptive':summary(all_acc,task),
        'policy_recorded_evidence_strata':recorded['evidence_strata'],
        'interpretation':'policy operating point selected on policy partition; whole-validation calibration diagnostics include calibration-fit patients'}
    subsets=[('whole_validation',np.ones(count,dtype=bool)),('calibration_partition',arrays['role']==0),('policy_partition',selected),
             ('physiology_observed',arrays['vital']==1),('no_physiology',arrays['vital']==0),('no_primitive',arrays['primitive']==0)]
    for name,mask in subsets:
        print(f'Computing saved-score discrimination: {name}',flush=True)
        report['discrimination'][name]=stream_metrics(arrays['labels'][mask],arrays['scores'][mask],b)
        report['discrimination'][name]['supported_patient_days']=int(mask.sum())*task.grid_minutes/1440
    original=recorded['window_diagnostics']['all']
    computed=report['discrimination']['policy_partition']
    for field in ('ap','auroc'): close(computed[field],original['raw'][field])
    for name in ('raw','calibrated'): close(computed['probabilities'][name],{k:original[name][k] for k in ('mean_risk','log_loss','brier')})
    report['seconds']=time.perf_counter()-begun
    write_json(output/'validation_results.json',report)
    evaluation={'version':'r0_test_evaluation_v1','status':'frozen_after_validation_audit',
        'model_complete_sha256':digest(MODEL/'complete.json'),'ssl_weights_sha256':digest(MODEL/'R0-ssl.pt'),
        'risk_weights_sha256':digest(MODEL/'R0-risk.pt'),'head_fit_sha256':digest(MODEL/'head_fit.json'),
        'scaler_sha256':digest(MODEL/'scaler.json'),'spec_sha256':fingerprint(spec),
        'task_identifier':task.identifier,'canonical_run':spec['canonical_run'],'dataset_sha256':spec['run_metadata_sha256'],
        'calibration':fitted,'threshold':cutoff,'alert_policy':spec['validation'],
        'metric_source_sha256':read(MODEL/'provenance.json')['source_sha256'],
        'metric_definitions':{'window':'unweighted AP, tie-aware AUROC, log loss, Brier on natural eligible stream',
            'episodes':'unique episodes with eligible forecast opportunities; one-to-one matching; no point adjustment',
            'burden':'unmatched emitted alerts divided by eligible grid-cell patient-days',
            'lead_time':'alarm initiation minus emitted prediction time',
            'evidence':'physiology observed versus no physiology; no-primitive descriptive when available'},
        'validation_scores_complete_sha256':digest(SCORES/'complete.json'),
        'validation_policy_complete_sha256':digest(POLICY/'complete.json'),
        'validation_audit_sha256':digest(output/'validation_results.json'),
        'software':read(MODEL/'provenance.json'),
        'audit_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'approved_scorer':'frozen CPU reference batch64 threads4; acceleration not yet approved for test',
        'test_execution':'not performed; requires separate manual authorization/implementation'}
    write_json(output/'final_evaluation_spec.json',evaluation)
    seal(output,purpose='validation independently audited; final evaluation specification frozen; no test execution')
    print('Validation audit complete; final_evaluation_spec.json frozen. No test evaluation executed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); audit(args.output)
