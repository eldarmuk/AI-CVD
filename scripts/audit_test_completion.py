"""Manual read-only full held-out artifact audit; no inference, fitting or tuning."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from scripts import test_evaluation as ev
from scripts.audit_validation import close
from src.final_study.config import verify_seal, seal


def console_text(path):
    data=path.read_bytes()
    if data.startswith((b'\xff\xfe',b'\xfe\xff')): return data.decode('utf-16')
    if b'\x00' in data[:100]: return data.decode('utf-16-le')
    return data.decode('utf-8-sig')


def metadata():
    current,e,task,_,_,root,blind=ev.preflight()
    out=ev.OUTPUT; marker=ev.read(out/'complete.json')
    if marker['status']!='complete': raise ValueError('Evaluation incomplete')
    for name in ('final_result_manifest.json','started.json','patients.json','recovery-0001.json'):
        ev.check_hash(out/name,marker['artifacts_sha256'][name])
    final=ev.read(out/'final_result_manifest.json')
    original=ev.read(ev.BASE/'r0-final-v1-test-preflight-v1.json')
    if final['preflight']!=original or ev.read(out/'started.json')['preflight']!=original:
        raise ValueError('Original receipt changed')
    if final['evaluation_spec']!=e or final['engineering_supplement']!=ev.read(ev.ENGINEERING) or final['source_sha256']!=ev.code_hashes():
        raise ValueError('Frozen model/scoring/metric binding changed')
    ignored={'git_commit','source_sha256'}
    if {k:v for k,v in current.items() if k not in ignored}!={k:v for k,v in original.items() if k not in ignored}:
        raise ValueError('Canonical population/task binding changed')
    patients=ev.read(out/'patients.json')
    if len(patients)!=current['patient_count'] or sum(p['samples'] for p in patients)!=current['windows'] or [p['patient'] for p in patients]!=current['patients']:
        raise ValueError('Patient manifest alignment changed')
    recoveries=sorted(out.glob('recovery-*.json'))
    for i,path in enumerate(recoveries,1):
        if path.name!=f'recovery-{i:04d}.json': raise ValueError('Recovery sequence changed')
        ev.check_hash(path,marker['artifacts_sha256'][path.name]); recovery=ev.read(path)
        if recovery['original_preflight']!=original or recovery['current_preflight']['source_sha256']!=final['source_sha256']:
            raise ValueError('Recovery binding mismatch')
        audit=recovery['audit']
        if audit['evaluation_spec_sha256']!=ev.SPEC_HASH or audit['engineering_sha256']!=ev.ENGINEERING_HASH:
            raise ValueError('Recovery scientific bindings changed')
        for name,h in audit['snapshot_sha256'].items():
            if name not in {f'{k}.npy' for k in ('scores','labels','vital','primitive','patient_index')}:
                if marker['artifacts_sha256'].get(name)!=h: raise ValueError('Recovered prefix changed')
    for name,h in final['artifacts_sha256'].items():
        if marker['artifacts_sha256'].get(name)!=h: raise ValueError('Final manifests disagree')
    log=ev.BASE/'r0-final-v1-test-recovery-v1.console.log'
    if 'TEST EVALUATION COMPLETE' not in console_text(log): raise ValueError('Missing completion log message')
    return current,e,task,root,blind,patients,marker,log


def audit(output):
    if output.exists() or not output.resolve().is_relative_to(ev.BASE.resolve()) or output.resolve().is_relative_to(ev.OUTPUT.resolve()):
        raise ValueError('Use a new immutable private audit directory outside the test run')
    begun=time.perf_counter(); current,e,task,root,blind,patients,marker,log=metadata()
    verify_seal(ev.OUTPUT)
    expected=set(marker['artifacts_sha256'])|{'complete.json'}
    actual={p.name for p in ev.OUTPUT.iterdir() if p.is_file()}
    if actual!=expected: raise ValueError('Unsealed or missing root artifacts')
    output.mkdir()
    input_marker=ev.digest(ev.OUTPUT/'complete.json')
    ev.write_json(output/'started.json',{'purpose':'read-only final artifact audit; no scoring or fitting',
        'test_complete_sha256':input_marker,'source_sha256':ev.digest(Path(__file__)),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ev.ROOT,text=True).strip()})
    export=ev.read(root/'exports/test-stream.json')
    ev.check_hash(root/'exports/test-stream.json',current['test_export_sha256'])
    pooled={k:np.load(ev.OUTPUT/f'{k}.npy',mmap_mode='r') for k in ('scores','labels','vital','primitive','patient_index')}
    for a in pooled.values():
        if a.shape!=(current['windows'],): raise ValueError('Pooled array length mismatch')
    accum={k:ev.accumulator() for k in ('all','physiology_observed','no_physiology','no_primitive')}
    position=0
    for ordinal,(row,s) in enumerate(zip(patients,export['shards'])):
        if any(s[k]!=v for k,v in blind['shards'][ordinal].items()): raise ValueError('Canonical shard metadata changed')
        if ev.read(ev.OUTPUT/f'patient-{ordinal:06d}.json')!=row: raise ValueError('Patient record/manifest mismatch')
        expected={'patient':s['senior_id'],'file':f'patient-{ordinal:06d}.npz','samples':s['samples'],
                  'source_shard_sha256':s['sha256'],'events':s['events'],'offset':position}
        if any(row[k]!=v for k,v in expected.items()): raise ValueError('Patient order/source mismatch')
        ev.check_hash(root/s['file'],s['sha256'])
        ev.check_hash(ev.OUTPUT/row['file'],row['sha256'])
        with np.load(root/s['file'],allow_pickle=False) as z:
            source={k:z[k] for k in ('features','end_rows','grid_start_ns','sample_ids','target','first_event','event_stop')}
        ends,times,ids=ev.verify_rows(row['patient'],source,s['events'],row['samples'],task)
        with np.load(ev.OUTPUT/row['file'],allow_pickle=False) as z: d={k:z[k] for k in z.files}
        vital,primitive=ev.evidence(source['features'],ends,task)
        values={'sample_ids':ids,'times':times,'labels':source['target'],'first_event':source['first_event'],
                'event_stop':source['event_stop'],'vital':vital,'primitive':primitive}
        if set(d)!=set(values)|{'scores','calibrated_scores'}: raise ValueError('Prediction schema changed')
        for k,v in values.items(): np.testing.assert_array_equal(d[k],v)
        raw=d['scores']
        if raw.shape!=times.shape or not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)): raise ValueError('Invalid scores')
        np.testing.assert_array_equal(d['calibrated_scores'],ev.calibrated(raw,e['calibration']['intercept']))
        for k in pooled:
            np.testing.assert_array_equal(pooled[k][position:position+len(times)],ordinal if k=='patient_index' else d[k])
        guards=row['guards']
        for k,v in {'samples':len(times),'batch':256,'reference_batch':64,'atol':1e-10,'rtol':1e-5}.items():
            if guards[k]!=v: raise ValueError('Scorer guard configuration changed')
        if guards['guarded_reference_batches']<min(2,(len(times)+63)//64): raise ValueError('Missing reference canaries')
        result=ev.fast_alerts(times,d['calibrated_scores'],s['events'],e['threshold'],task,e['alert_policy']['cooldown_minutes'],vital)
        if result!=row['operational']: raise ValueError('Policy replay mismatch')
        ev.add(accum['all'],result)
        for name,mask in [('physiology_observed',vital),('no_physiology',~vital),('no_primitive',~primitive)]:
            sub=ev.subgroup_result(result,times,mask,s['events'],task)
            if sub!=row['strata'][name]: raise ValueError('Subgroup attribution mismatch')
            if sub['supported_cells']: ev.add(accum[name],sub)
        position+=len(times)
        if (ordinal+1)%100==0: print(f'Audited saved test patient {ordinal+1}/{len(patients)}',flush=True)
    if position!=current['windows'] or len(export['shards'])!=len(patients): raise ValueError('Missing samples')
    # Only now compare saved aggregate metrics; no scientific choices are made.
    saved=ev.read(ev.OUTPUT/'test_results.json')
    if saved['status']!='complete' or saved['calibration']!=e['calibration'] or saved['threshold']!=e['threshold'] or saved['alert_policy']!=e['alert_policy'] or saved['metric_definitions']!=e['metric_definitions']:
        raise ValueError('Saved scientific result specification changed')
    for name,a in accum.items(): close(saved['operational'][name],ev.summary(a,task))
    for name,mask in [('all',slice(None)),('physiology_observed',pooled['vital']),('no_physiology',~pooled['vital']),('no_primitive',~pooled['primitive'])]:
        print(f'Checking saved-score metric arithmetic: {name}',flush=True)
        close(saved['discrimination'][name],ev.stream_metrics(pooled['labels'][mask],pooled['scores'][mask],e['calibration']['intercept']))
    if saved['fraction_eligible_supported_time_scored']!=1.: raise ValueError('Coverage mismatch')
    ev.check_hash(ev.OUTPUT/'complete.json',input_marker)
    ev.write_json(output/'audit_results.json',{'status':'passed','patients':len(patients),'windows':position,
        'test_complete_sha256':input_marker,'test_results_sha256':ev.digest(ev.OUTPUT/'test_results.json'),
        'console_sha256':ev.digest(log),'frozen_spec_sha256':ev.SPEC_HASH,'engineering_sha256':ev.ENGINEERING_HASH,
        'checks':'all artifact hashes; exact canonical IDs/labels/order; pooled alignment; source hashes; calibration; policy and metric replay; unchanged recovery prefix',
        'inference_executed':False,'fitting_executed':False,'seconds':time.perf_counter()-begun,
        'limitation':'Scores authenticated by original sealed hashes and frozen scorer provenance, not recomputed by the model.'})
    seal(output,purpose='completed held-out result audit; no model inference or fitting')
    print('TEST RESULT AUDIT COMPLETE: audit_results.json passed; complete.json sealed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); audit(args.output)
