"""Manual natural-stream canonical VALIDATION scoring and operating-point development."""
import json
from pathlib import Path
import numpy as np
import torch
from src.ai_cvd.features import FEATURE_NAMES
from src.architecture_study.config import ROOT, fingerprint
from src.architecture_study.data import TrainingStore, digest, write_json
from src.architecture_study.projection import project
from .training import verify_run
from .config import seal, verify_seal, runtime
from .policy import authorize_validation, calibration_partition, fit_intercept, select_threshold, evaluate, calibrated
from src.architecture_study.analyze import metrics


def score(spec,c,task,model_run,output):
    if output.exists():
        raise FileExistsError('Immutable validation output exists')
    model,scaler = verify_run(model_run,spec,c)
    torch.set_num_threads(spec['cpu_threads']); torch.use_deterministic_algorithms(True)
    root = ROOT/spec['canonical_run']
    meta = json.loads((root/'run_metadata.json').read_text())
    if digest(root/'run_metadata.json') != spec['run_metadata_sha256']:
        raise ValueError('Canonical metadata changed')
    name = 'exports/validation-stream.json'
    if digest(root/name) != meta['artifacts_sha256'][name]:
        raise ValueError('Validation export changed')
    export = json.loads((root/name).read_text())
    if tuple(export['feature_names']) != FEATURE_NAMES or export['task_identifier'] != task.identifier:
        raise ValueError('Validation schema mismatch')
    patients = [s['senior_id'] for s in export['shards']]
    authorize_validation('validation',patients,task)
    if len(set(patients)) != len(patients) or any(s['split'] != 'validation' for s in export['shards']):
        raise ValueError('Duplicate/misassigned validation patient')
    output.mkdir(parents=True)
    provenance = {'runtime':runtime(),'split':'validation','spec_sha256':fingerprint(spec),'model_complete_sha256':digest(model_run/'complete.json'),
        'run_metadata_sha256':spec['run_metadata_sha256'],'export_sha256':digest(root/name),
        'feature_order':FEATURE_NAMES,'task_identifier':task.identifier,'patient_ids':patients,
        'population':'every eligible indexed stream window; no label-dependent subsampling'}
    write_json(output/'provenance.json',provenance)
    manifest = []
    # Identity reconstruction reuses the canonical TRAIN reader's pure identity method;
    # no training reader/population is repurposed or authorized for validation fitting.
    identity = type('IdentityContext',(),{'task':task})()
    for number,shard in enumerate(export['shards']):
        p = shard['senior_id']; path = (root/shard['file']).resolve()
        if not path.is_relative_to((root/'patients').resolve()) or digest(path) != shard['sha256']:
            raise ValueError('Validation patient shard mismatch')
        with np.load(path,allow_pickle=False) as z:
            a = {k:z[k] for k in ('features','end_rows','sample_ids','grid_start_ns','target','first_event','event_stop')}
        n = len(a['end_rows'])
        if n != shard['samples'] or a['features'].shape[1] != len(FEATURE_NAMES) or np.any(np.diff(a['end_rows'])<=0) or np.any(a['end_rows']<task.sequence_steps-1) or np.any(a['end_rows']>=len(a['features'])):
            raise ValueError('Invalid validation indexing')
        ids = []; times = []; scores = []; evidence = []
        for start in range(0,n,spec['batch_size']):
            stop = min(n,start+spec['batch_size']); windows = []
            for i in range(start,stop):
                sid,t = TrainingStore.identity(identity,p,i,a)
                ids.append(sid); times.append(t)
                end = int(a['end_rows'][i]); windows.append(a['features'][end-task.sequence_steps+1:end+1])
                linked = shard['events'][int(a['first_event'][i]):int(a['event_stop'][i])]
                if bool(linked) != bool(a['target'][i]) or any(not t < e[0] <= t+task.horizon_minutes*60*10**9 for e in linked):
                    raise ValueError('Validation outcome alignment mismatch')
            view = project(torch.tensor(np.stack(windows)),scaler['mean'],scaler['scale'],task)
            with torch.no_grad():
                prediction = model(view).sigmoid().flatten().numpy()
                # First batch for each patient is a deterministic repeated-inference check.
                if start == 0 and not np.array_equal(prediction,model(view).sigmoid().flatten().numpy()):
                    raise ValueError('Nondeterministic inference')
            scores.extend(prediction.tolist()); evidence.extend(view.M[:,:,:5].any((1,2)).tolist())
        if len(set(ids)) != n:
            raise ValueError('Duplicate validation sample')
        filename = f'patient-{number:06d}.npz'
        np.savez_compressed(output/filename,sample_ids=np.asarray(ids),times=np.asarray(times,dtype=np.int64),
                            scores=np.asarray(scores),vital=np.asarray(evidence),labels=a['target'])
        manifest.append({'patient':p,'file':filename,'sha256':digest(output/filename),'events':shard['events'],
                         'samples':n,'source_shard_sha256':shard['sha256']})
        print(f'Validation patient {number+1}/{len(patients)} scored',flush=True)
    write_json(output/'patients.json',manifest)
    seal(output,purpose='canonical validation stream predictions; no test access')


def develop(spec,c,task,model_run,predictions,output):
    if output.exists():
        raise FileExistsError('Immutable validation development output exists')
    verify_run(model_run,spec,c); verify_seal(predictions)
    provenance = json.loads((predictions/'provenance.json').read_text())
    if provenance['split'] != 'validation' or provenance['spec_sha256'] != fingerprint(spec) or provenance['model_complete_sha256'] != digest(model_run/'complete.json'):
        raise ValueError('Stale/non-validation predictions')
    manifest = json.loads((predictions/'patients.json').read_text())
    authorize_validation('validation',[r['patient'] for r in manifest],task)
    if [r['patient'] for r in manifest] != provenance['patient_ids']:
        raise ValueError('Validation patient manifest changed')
    calibration = {}; policy = {}
    for r in manifest:
        path = predictions/r['file']
        if not path.resolve().is_relative_to(predictions.resolve()) or digest(path) != r['sha256']:
            raise ValueError('Prediction hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            data = {k:z[k] for k in ('scores','times','vital','labels','sample_ids')}
        if len(set(data['sample_ids'])) != r['samples'] or any(len(v)!=r['samples'] for v in data.values()):
            raise ValueError('Prediction alignment mismatch')
        data['events'] = r['events']
        target = calibration if calibration_partition(r['patient'],spec['validation']['seed']) else policy
        target[r['patient']] = data
    if not calibration or not policy:
        raise ValueError('Empty validation development partition')
    y = np.concatenate([r['labels'] for r in calibration.values()]); p = np.concatenate([r['scores'] for r in calibration.values()])
    patients = [sid for sid,r in calibration.items() for _ in r['labels']]
    fit = fit_intercept(p,y,split='validation',patients=patients,task=task)
    if fit.get('boundary_hit'):
        raise ValueError('Calibration reached intercept bound; no finalized operating point')
    threshold = select_threshold(policy,fit['intercept'],split='validation',task=task,config=spec['validation'])
    result = evaluate(policy,threshold['selected']['threshold'],task,spec['validation']['cooldown_minutes'],fit['intercept'])
    result['fraction_supported_time_scored'] = 1.0  # All eligible cells were verified against the export.
    result['coverage_interpretation'] = 'all canonical eligible cells scored; unsupported gaps are excluded, not treated as negative'
    policy_y = np.concatenate([d['labels'] for d in policy.values()])
    policy_raw = np.concatenate([d['scores'] for d in policy.values()])
    policy_vital = np.concatenate([d['vital'] for d in policy.values()]).astype(bool)
    result['window_diagnostics'] = {}
    for name,mask in [('all',np.ones(len(policy_y),dtype=bool)),('physiology_observed',policy_vital),('no_physiology',~policy_vital)]:
        result['window_diagnostics'][name] = {
            'raw':metrics(policy_y[mask],policy_raw[mask],np.ones(int(mask.sum()))),
            'calibrated':metrics(policy_y[mask],calibrated(policy_raw[mask],fit['intercept']),np.ones(int(mask.sum())))}
    output.mkdir(parents=True)
    write_json(output/'calibration.json',fit); write_json(output/'threshold.json',threshold)
    write_json(output/'policy_development_metrics.json',result)
    write_json(output/'provenance.json',{'runtime':runtime(),'spec_sha256':fingerprint(spec),'model_complete_sha256':digest(model_run/'complete.json'),
        'prediction_complete_sha256':digest(predictions/'complete.json'),'calibration_patients':sorted(calibration),
        'policy_patients':sorted(policy),'policy_parameters':spec['validation'],
        'interpretation':'validation development; operating point selected here, not an unbiased final estimate',
        'sample_ids_bound_by':'hashed complete validation prediction manifest; all rows in each patient partition'})
    seal(output,purpose='frozen validation intercept/threshold; ready for separate Problem 10 review')
    print('Validation development complete; no test evaluation performed.',flush=True)
