"""Validation-only audit helpers, outside the frozen training source tree."""
import json
import numpy as np
from src.architecture_study.config import ROOT,fingerprint
from src.architecture_study.data import digest
from src.final_study.config import load,verify_seal,source_hashes
from src.final_study.training import verify_run
from src.final_study.policy import authorize_validation,calibration_partition

BASE=ROOT/'runs/final_studies'
MODEL=BASE/'r0-final-v1'
SCORES=BASE/'r0-final-v1-validation-scores'
POLICY=BASE/'r0-final-v1-validation-policy'


def read(path):
    return json.loads(path.read_text())


def metadata():
    spec,c,task=load(); model,scaler=verify_run(MODEL,spec,c)
    verify_seal(POLICY)
    marker=read(SCORES/'complete.json')
    if marker['status']!='complete': raise ValueError('Scoring incomplete')
    for name in ('patients.json','provenance.json'):
        if digest(SCORES/name)!=marker['artifacts_sha256'][name]: raise ValueError('Score metadata changed')
    sp=read(SCORES/'provenance.json'); pp=read(POLICY/'provenance.json')
    manifest=read(SCORES/'patients.json')
    root=ROOT/spec['canonical_run']; runmeta=read(root/'run_metadata.json')
    if digest(root/'run_metadata.json')!=spec['run_metadata_sha256']: raise ValueError('Canonical run changed')
    export_path=root/'exports/validation-stream.json'
    if digest(export_path)!=runmeta['artifacts_sha256']['exports/validation-stream.json'] or digest(export_path)!=sp['export_sha256']:
        raise ValueError('Validation export changed')
    export=read(export_path)
    if export['task_identifier']!=task.identifier or export['feature_names']!=sp['feature_order']:
        raise ValueError('Validation schema changed')
    for p in (sp,pp):
        if p['spec_sha256']!=fingerprint(spec) or p['model_complete_sha256']!=digest(MODEL/'complete.json') or p['runtime']['source_sha256']!=source_hashes():
            raise ValueError('Frozen model/source binding changed')
    if sp['split']!='validation' or pp['prediction_complete_sha256']!=digest(SCORES/'complete.json') or pp['policy_parameters']!=spec['validation']:
        raise ValueError('Validation provenance changed')
    patients=[r['patient'] for r in manifest]; authorize_validation('validation',patients,task)
    if len(set(patients))!=len(patients) or patients!=sp['patient_ids'] or patients!=[s['senior_id'] for s in export['shards']]:
        raise ValueError('Validation population/order mismatch')
    train=read(MODEL/'patients.json')
    if set(patients)&set(train['fit']+train['held']): raise ValueError('Training/validation overlap')
    cal=sorted(p for p in patients if calibration_partition(p,spec['validation']['seed']))
    pol=sorted(set(patients)-set(cal))
    if pp['calibration_patients']!=cal or pp['policy_patients']!=pol or set(cal)&set(pol):
        raise ValueError('Validation role mismatch')
    if set(marker['artifacts_sha256'])!={'patients.json','provenance.json'}|{r['file'] for r in manifest}:
        raise ValueError('Incomplete scoring hash manifest')
    for r,s in zip(manifest,export['shards']):
        if s['split']!='validation' or r['samples']!=s['samples'] or r['events']!=s['events'] or r['source_shard_sha256']!=s['sha256'] or r['sha256']!=marker['artifacts_sha256'][r['file']]:
            raise ValueError('Patient manifest/source link mismatch')
    threshold=read(POLICY/'threshold.json')
    cap=spec['validation']['false_alerts_per_supported_day_cap']
    eligible=[r for r in threshold['candidates'] if r['false_alerts_per_supported_day']<=cap]
    best=max(eligible,key=lambda r:(r['episode_sensitivity'],r['alert_precision'] or 0.,r['threshold']))
    if best!=threshold['selected'] or threshold['criterion']!=spec['validation']['threshold_objective']:
        raise ValueError('Recorded threshold choice violates frozen rule')
    return spec,c,task,model,scaler,root,manifest,export,cal,pol


def vector_windows(features,ends,steps):
    ends=np.asarray(ends,dtype=np.int64)
    if np.any(ends<steps-1) or np.any(ends>=len(features)): raise ValueError('Invalid prefix bounds')
    return np.ascontiguousarray(features[ends[:,None]-np.arange(steps-1,-1,-1)[None,:]])
