"""Small fixed VALIDATION-only equivalence gate; never runs test."""
import argparse
import json
import time
import subprocess
from pathlib import Path
import numpy as np
from scripts.validation_support import metadata,read,SCORES,POLICY,MODEL
from scripts.accelerated_r0 import score_patient,check_equivalence,ATOL,RTOL
from src.architecture_study.config import ROOT
from src.architecture_study.data import digest,write_json
from src.final_study.config import verify_seal
from src.final_study.policy import calibrated,patient_alerts


def approve(output):
    if output.exists() or not output.resolve().is_relative_to((ROOT/'runs').resolve()): raise ValueError('Use new private supplement path')
    audit=ROOT/'runs/final_studies/r0-final-v1-validation-audit'
    verify_seal(audit)
    spec,c,task,model,scaler,root,manifest,export,cal,pol=metadata()
    evaluation=read(audit/'final_evaluation_spec.json')
    if evaluation['validation_audit_sha256']!=digest(audit/'validation_results.json') or evaluation['model_complete_sha256']!=digest(MODEL/'complete.json'):
        raise ValueError('Evaluation binding mismatch')
    threshold=evaluation['threshold']; intercept=evaluation['calibration']['intercept']
    rows=[]; start=time.perf_counter()
    # Expand the same first-three-patient benchmark to full patient trajectories,
    # fixed before checking outcomes; no population/threshold selection.
    for r,s in zip(manifest[:3],export['shards'][:3]):
        if digest(root/s['file'])!=s['sha256'] or digest(SCORES/r['file'])!=r['sha256']: raise ValueError('Subset artifacts changed')
        with np.load(root/s['file'],allow_pickle=False) as z:
            features=z['features']; ends=z['end_rows']; ids=z['sample_ids']; origin=int(z['grid_start_ns'])
        with np.load(SCORES/r['file'],allow_pickle=False) as z:
            original=z['scores']; times=z['times']; vital=z['vital']; saved_ids=z['sample_ids'].tolist()
        if saved_ids!=['sample_'+bytes(v).hex() for v in ids]: raise ValueError('Sample ordering differs')
        np.testing.assert_array_equal(times,origin+(ends.astype(np.int64)+1)*task.grid_minutes*60*10**9)
        begun=time.perf_counter()
        fast,guards=score_patient(model,features,ends,scaler,task,threshold,intercept)
        seconds=time.perf_counter()-begun
        repeated,_=score_patient(model,features,ends,scaler,task,threshold,intercept)
        if not np.array_equal(fast,repeated): raise ValueError('Accelerated inference not deterministic')
        check_equivalence(fast,original)
        raw_decisions=calibrated(original,intercept)>=threshold
        fast_decisions=calibrated(fast,intercept)>=threshold
        np.testing.assert_array_equal(raw_decisions,fast_decisions)
        a=patient_alerts(times,calibrated(original,intercept),r['events'],threshold,task,spec['validation']['cooldown_minutes'],vital)
        b=patient_alerts(times,calibrated(fast,intercept),r['events'],threshold,task,spec['validation']['cooldown_minutes'],vital)
        if a!=b: raise ValueError('Downstream policy changed')
        rows.append({'patient':r['patient'],'samples':len(ends),'sample_order_sha256':digest(SCORES/r['file']),
            'seconds':seconds,'examples_per_second':len(ends)/seconds,'max_absolute_error':float(np.max(abs(fast-original))),
            'alerts':len(a['alerts']),'matched_alerts':len(a['matched_episodes']),'guards':guards})
        print(f'Validated fixed patient trajectory {len(rows)}/3',flush=True)
    supplement={'status':'approved_engineering_adapter','version':'cpu256_guarded_v1',
        'final_evaluation_spec_sha256':digest(audit/'final_evaluation_spec.json'),
        'model_complete_sha256':digest(MODEL/'complete.json'),'risk_weights_sha256':digest(MODEL/'R0-risk.pt'),
        'intercept':intercept,'threshold':threshold,'device':'cpu','threads':4,'batch_size':256,
        'reference_batch_size':64,'numeric_tolerance':{'absolute':ATOL,'relative':RTOL},
        'guard_rule':'first/last reference batch per patient plus all batches within twice tolerance of inverse-calibrated threshold; replay and replace exact reference scores; fail on discrepancy',
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/accelerated_r0.py',ROOT/'scripts/validation_support.py']},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'equivalence_scope':'first three complete validation patient trajectories, fixed independently of labels',
        'rows':rows,'wall_seconds':time.perf_counter()-start,
        'limitation':'finite-subset numerical and exact decision equivalence; not a proof of bitwise equivalence on unseen samples',
        'test_requirements':'Problem 10 runner must enforce all bindings, guards and deterministic settings; fail closed rather than retune; no test runner or test execution in this approval'}
    write_json(output,supplement)
    print(json.dumps({k:v for k,v in supplement.items() if k not in ('rows','source_sha256')},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='patient'} for r in rows],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();approve(args.output)
