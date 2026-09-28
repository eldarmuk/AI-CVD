"""Problem 10: fixed, evaluation-only manual entry point. No fitting or resume."""
import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from scripts.accelerated_r0 import score_patient
from scripts.validation_audit_math import fast_alerts, accumulator, add, summary, stream_metrics
from src.architecture_study.config import ROOT, fingerprint
from src.architecture_study.data import digest, write_json
from src.architecture_study.artifacts import load_checkpoint
from src.architecture_study.projection import PRIMITIVES
from src.ai_cvd.features import FEATURE_NAMES
from src.ai_cvd.dataset import patient_split, sample_id
from src.final_study.config import load, verify_seal, seal, source_hashes
from src.final_study.policy import calibrated

BASE = ROOT/'runs/final_studies'
MODEL = BASE/'r0-final-v1'
AUDIT = BASE/'r0-final-v1-validation-audit'
SPEC = AUDIT/'final_evaluation_spec.json'
ENGINEERING = BASE/'r0-final-evaluation-engineering-v1.json'
OUTPUT = BASE/'r0-final-v1-test'
SPEC_HASH = '0ad2c76b2f8ec65ead77c66067e3760d3581e6d762b47a1ed2e3cb5a6986899c'
ENGINEERING_HASH = 'd284a32cbf27a315f72cd8a8da8dd7bd4cd622cadf3ccae19d831221e3e5be83'
CODE = ('scripts/test_evaluation.py', 'scripts/test_uncertainty.py',
        'scripts/validation_audit_math.py', 'scripts/accelerated_r0.py', 'scripts/validation_support.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Frozen hash mismatch: {path.name}')


def selected_json(path, fields):
    """Lexically skip unrequested values, without decoding outcome values.

    Byte access is needed for hashing/structural scanning of mixed metadata.
    Only whitelisted object fields are decoded. Lists recurse using the same
    field selection. None means an explicitly authorized complete value.
    """
    text = path.read_text(encoding='utf-8'); dec = json.JSONDecoder()
    def ws(i):
        while i < len(text) and text[i].isspace(): i += 1
        return i
    def skip(i):
        i = ws(i); start = i; depth = 0; string = False; escape = False
        while i < len(text):
            c = text[i]
            if string:
                if escape: escape = False
                elif c == '\\': escape = True
                elif c == '"': string = False
            elif c == '"': string = True
            elif c in '[{': depth += 1
            elif c in ']}':
                if depth == 0: break
                depth -= 1
                if depth == 0: return i+1
            elif c == ',' and depth == 0: break
            i += 1
        if i == start: raise ValueError('Invalid metadata JSON')
        return i
    def parse(i, wanted):
        i = ws(i)
        if wanted is None: return dec.raw_decode(text, i)
        if text[i] == '[':
            result = []; i = ws(i+1)
            while text[i] != ']':
                value, i = parse(i, wanted); result.append(value); i = ws(i)
                if text[i] == ',': i = ws(i+1)
                elif text[i] != ']': raise ValueError('Invalid list')
            return result, i+1
        if text[i] != '{': raise ValueError('Expected metadata object')
        result = {}; seen = set(); i = ws(i+1)
        while text[i] != '}':
            key, i = dec.raw_decode(text, i); i = ws(i)
            if key in seen or text[i] != ':': raise ValueError('Duplicate/invalid metadata key')
            seen.add(key); i = ws(i+1)
            if key in wanted: result[key], i = parse(i, wanted[key])
            else: i = skip(i)
            i = ws(i)
            if text[i] == ',': i = ws(i+1)
            elif text[i] != '}': raise ValueError('Invalid object')
        return result, i+1
    result, end = parse(0, fields)
    if ws(end) != len(text): raise ValueError('Trailing metadata')
    return result


def code_hashes():
    return {p:digest(ROOT/p) for p in CODE}


def preflight():
    check_hash(SPEC, SPEC_HASH); check_hash(ENGINEERING, ENGINEERING_HASH)
    check_hash(ROOT/'scripts/validation_audit_math.py',
               '3b2108fe77e638db363079cae711f2852ccabbe66593e3cc32bad0ec52f55759')
    e = read(SPEC); engineering = read(ENGINEERING)
    spec, config, task = load()
    if e['spec_sha256'] != fingerprint(spec) or e['task_identifier'] != task.identifier:
        raise ValueError('Task/config mismatch')
    if e['metric_source_sha256'] != source_hashes(): raise ValueError('Frozen source changed')
    for p,h in engineering['source_sha256'].items(): check_hash(ROOT/p, h)
    if engineering['final_evaluation_spec_sha256'] != SPEC_HASH or engineering['threshold'] != e['threshold'] or engineering['intercept'] != e['calibration']['intercept']:
        raise ValueError('Engineering/scientific binding changed')
    for name,key in [('complete.json','model_complete_sha256'),('R0-ssl.pt','ssl_weights_sha256'),
                     ('R0-risk.pt','risk_weights_sha256'),('head_fit.json','head_fit_sha256'),('scaler.json','scaler_sha256')]:
        check_hash(MODEL/name, e[key])
    verify_seal(MODEL)
    model_meta = read(MODEL/'provenance.json')
    if model_meta['config_sha256'] != fingerprint(config) or model_meta['source_sha256'] != source_hashes():
        raise ValueError('Model provenance changed')
    if platform.python_version() != e['software']['python'] or any(version(k) != v for k,v in e['software']['packages'].items()):
        raise ValueError('Unapproved Python/package environment')
    check_hash(AUDIT/'validation_results.json',e['validation_audit_sha256'])
    for folder,key in [('r0-final-v1-validation-scores','validation_scores_complete_sha256'),
                       ('r0-final-v1-validation-policy','validation_policy_complete_sha256')]:
        check_hash(BASE/folder/'complete.json',e[key])
    root = ROOT/e['canonical_run']; check_hash(root/'run_metadata.json',e['dataset_sha256'])
    meta = selected_json(root/'run_metadata.json', {'artifacts_sha256':None})
    export_path = root/'exports/test-stream.json'
    check_hash(export_path,meta['artifacts_sha256']['exports/test-stream.json'])
    fields = {'senior_id':None,'file':None,'sha256':None,'samples':None,'split':None}
    export = selected_json(export_path, {'task_identifier':None,'feature_names':None,'shards':fields})
    if export['task_identifier'] != task.identifier or tuple(export['feature_names']) != FEATURE_NAMES or list(FEATURE_NAMES) != e['software']['feature_order']:
        raise ValueError('Export task/projection changed')
    patients = [s['senior_id'] for s in export['shards']]
    if len(set(patients)) != len(patients) or any(patient_split(p,task) != 'test' for p in patients):
        raise ValueError('Invalid test membership')
    train = read(MODEL/'patients.json')
    validation = read(BASE/'r0-final-v1-validation-scores/patients.json')
    if set(patients) & (set(train['fit']+train['held']) | {r['patient'] for r in validation}):
        raise ValueError('Patient isolation failure')
    for s in export['shards']:
        if s['split'] != 'test' or s['samples'] <= 0 or not (root/s['file']).resolve().is_relative_to((root/'patients').resolve()):
            raise ValueError('Invalid shard metadata')
    # Loading weights is not inference and does not touch any test shard.
    scaler = read(MODEL/'scaler.json')
    model = load_checkpoint(MODEL/'R0-risk.pt','R0',config,scaler,'risk')
    model.requires_grad_(False); model.eval()
    receipt = {'status':'passed_outcome_blind_preflight','evaluation_spec_sha256':SPEC_HASH,
        'engineering_sha256':ENGINEERING_HASH,'dataset_sha256':e['dataset_sha256'],
        'test_export_sha256':digest(export_path),'patients':patients,'patient_count':len(patients),
        'windows':sum(s['samples'] for s in export['shards']), 'source_sha256':code_hashes(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'outcome_access':False,'shards_opened':False,
        'limitations':'Mixed metadata bytes hashed/scanned; event values not decoded. Per-row labels, IDs, shard contents and outcome linkage remain execution-time checks.'}
    return receipt,e,task,model,scaler,root,export


def reserve(output):
    # Fixed output path also prevents rerunning under a different directory name.
    if output.resolve() != OUTPUT.resolve(): raise ValueError('Only the fixed one-shot output is permitted')
    output.mkdir(parents=True,exist_ok=False)


def evidence(features, ends, task):
    valid = np.isfinite(features[:,[FEATURE_NAMES.index(p) for p in PRIMITIVES]])
    result = []
    for width in (5,6):
        prefix = np.r_[0,np.cumsum(valid[:,:width].any(1))]
        result.append(prefix[ends+1]-prefix[ends-task.sequence_steps+1] > 0)
    return result


def verify_rows(patient, a, events, n, task):
    ends = a['end_rows'].astype(np.int64)
    if len(ends) != n or a['features'].shape[1] != len(FEATURE_NAMES) or np.any(np.diff(ends)<=0) or np.any(ends<task.sequence_steps-1) or np.any(ends>=len(a['features'])):
        raise ValueError('Invalid sequence bounds/order')
    times = int(a['grid_start_ns'])+(ends+1)*task.grid_minutes*60*10**9
    ids = ['sample_'+bytes(v).hex() for v in a['sample_ids']]
    if len(ids) != n or len(set(ids)) != n: raise ValueError('Duplicate/missing IDs')
    for sid,t in zip(ids,times):
        if sid != sample_id(patient,datetime.fromtimestamp(int(t)/1e9,timezone.utc)):
            raise ValueError('Sample ID/cutoff mismatch')
    if len({str(e) for _,e in events}) != len(events) or events != sorted(events):
        raise ValueError('Duplicate/unordered episodes')
    et = np.array([t for t,_ in events],dtype=np.int64)
    first = np.searchsorted(et,times,side='right')
    stop = np.searchsorted(et,times+task.horizon_minutes*60*10**9,side='right')
    for actual,expected in [(a['first_event'],first),(a['event_stop'],stop),(a['target'],stop>first)]:
        np.testing.assert_array_equal(actual,expected)
    return ends,times,np.asarray(ids)


def subgroup_result(result, times, mask, events, task):
    """Attribute existing alerts; never rerun crossing/cooldown on filtered time."""
    alerts = [a for a in result['alerts'] if mask[np.searchsorted(times,a['prediction_ns'])]]
    eligible = []
    subset = times[mask]
    for t,e in events:
        lo,hi = np.searchsorted(subset,[t-task.horizon_minutes*60*10**9,t],side='left')
        if lo<hi: eligible.append(e)
    return {'alerts':alerts,'eligible_episodes':eligible,
        'matched_episodes':[a['episode_id'] for a in alerts if a['episode_id'] is not None],
        'supported_cells':int(mask.sum()),'span_cells':result['span_cells']}


@torch.inference_mode()
def execute(receipt_path):
    if OUTPUT.exists(): raise FileExistsError('One-shot output exists: preserve partial/completed work; explicit recovery review required')
    receipt,e,task,model,scaler,root,blind = preflight()
    approved = read(receipt_path)
    if approved != receipt: raise ValueError('Preflight receipt stale; no test access')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit reviewed code before definitive evaluation')
    reserve(OUTPUT); begun = time.perf_counter()
    write_json(OUTPUT/'started.json',{'status':'started','utc':datetime.now(timezone.utc).isoformat(),
        'preflight':receipt,'python':platform.python_version(),
        'packages':{p:version(p) for p in ('torch','numpy','pandas','scipy')},
        'recovery':'No automatic resume/restart. Preserve all files for explicit recovery review.'})
    # First outcome access occurs only after every frozen gate and one-shot reservation.
    export = read(root/'exports/test-stream.json')
    check_hash(root/'exports/test-stream.json',receipt['test_export_sha256'])
    n = receipt['windows']; arrays = {key:np.lib.format.open_memmap(OUTPUT/f'{key}.npy',mode='w+',dtype=dtype,shape=(n,))
        for key,dtype in [('scores','float64'),('labels','uint8'),('vital','bool'),('primitive','bool'),('patient_index','int32')]}
    accum = {key:accumulator() for key in ('all','physiology_observed','no_physiology','no_primitive')}
    manifest = []; pos = 0
    for ordinal,s in enumerate(export['shards']):
        p = s['senior_id']; b = blind['shards'][ordinal]
        if any(s[k] != v for k,v in b.items()): raise ValueError('Blind manifest changed')
        path = root/s['file']; check_hash(path,s['sha256'])
        with np.load(path,allow_pickle=False) as z:
            a = {k:z[k] for k in ('features','end_rows','grid_start_ns','sample_ids','target','first_event','event_stop')}
        ends,times,ids = verify_rows(p,a,s['events'],s['samples'],task)
        raw,guards = score_patient(model,a['features'],ends,scaler,task,e['threshold'],e['calibration']['intercept'])
        if raw.shape != times.shape or any(q.requires_grad for q in model.parameters()): raise ValueError('Inference contract broken')
        pc = calibrated(raw,e['calibration']['intercept']); vital,primitive = evidence(a['features'],ends,task)
        result = fast_alerts(times,pc,s['events'],e['threshold'],task,e['alert_policy']['cooldown_minutes'],vital)
        add(accum['all'],result)
        strata = {}
        for name,mask in [('physiology_observed',vital),('no_physiology',~vital),('no_primitive',~primitive)]:
            strata[name] = subgroup_result(result,times,mask,s['events'],task)
            if strata[name]['supported_cells']: add(accum[name],strata[name])
        file = f'patient-{ordinal:06d}.npz'
        np.savez_compressed(OUTPUT/file,sample_ids=ids,times=times,scores=raw,calibrated_scores=pc,
            labels=a['target'],vital=vital,primitive=primitive,first_event=a['first_event'],event_stop=a['event_stop'])
        row = {'patient':p,'file':file,'sha256':digest(OUTPUT/file),'samples':len(times),
            'offset':pos,'source_shard_sha256':s['sha256'],'events':s['events'],
            'operational':result,'strata':strata,'guards':guards}
        manifest.append(row)
        write_json(OUTPUT/f'patient-{ordinal:06d}.json',row)
        for key,value in [('scores',raw),('labels',a['target']),('vital',vital),('primitive',primitive),('patient_index',ordinal)]:
            arrays[key][pos:pos+len(times)] = value
        pos += len(times)
        print(f'Test patient {ordinal+1}/{len(export["shards"])} complete; {pos}/{n} windows',flush=True)
    if pos != n: raise ValueError('Incomplete population')
    for a in arrays.values(): a.flush()
    report = {'status':'complete','calibration':e['calibration'],'threshold':e['threshold'],
        'alert_policy':e['alert_policy'],'discrimination':{},'operational':{k:summary(v,task) for k,v in accum.items()},
        'fraction_eligible_supported_time_scored':1.,'metric_definitions':e['metric_definitions'],
        'coverage_limitation':'Eligible support only, not verified enrollment or continuous clinical coverage.',
        'subgroup_limitation':'Episode opportunity sets overlap; subgroup hits attributed by emitted alert evidence. Do not add subgroup denominators.',
        'uncertainty':'Pending separate immutable patient-cluster bootstrap command; no model selection.'}
    for name,mask in [('all',slice(None)),('physiology_observed',arrays['vital']),('no_physiology',~arrays['vital']),('no_primitive',~arrays['primitive'])]:
        report['discrimination'][name] = stream_metrics(arrays['labels'][mask],arrays['scores'][mask],e['calibration']['intercept'])
    validation = read(AUDIT/'validation_results.json')
    report['frozen_validation_comparator'] = {'discrimination':validation['discrimination'],
        'operational':validation['whole_validation_operational_descriptive']}
    report['comparison_rule'] = 'Descriptive only; no significance or post-test retuning. Review discrimination, process dependence, episode detection, burden and lead time together.'
    report['seconds'] = time.perf_counter()-begun
    import psutil
    memory = psutil.Process().memory_info()
    report['resources'] = {'device':'cpu','threads':4,'batch_size':256,
        'peak_working_set_bytes':getattr(memory,'peak_wset',None),'current_rss_bytes':memory.rss,
        'end_to_end_windows_per_second':n/report['seconds']}
    write_json(OUTPUT/'patients.json',manifest); write_json(OUTPUT/'test_results.json',report)
    write_json(OUTPUT/'final_result_manifest.json',{'preflight':receipt,'evaluation_spec':e,
        'engineering_supplement':read(ENGINEERING),'source_sha256':code_hashes(),
        'artifacts_sha256':{str(p.relative_to(OUTPUT)):digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}})
    seal(OUTPUT,purpose='one-shot canonical test evaluation; no fitting; uncertainty separate')
    print('TEST EVALUATION COMPLETE: complete.json sealed; do not rerun.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('preflight'); p.add_argument('--receipt',type=Path,required=True)
    p = sub.add_parser('run'); p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--acknowledge-one-shot',action='store_true',required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        if not args.receipt.resolve().is_relative_to(BASE.resolve()): raise ValueError('Private receipt must stay under runs/final_studies')
        r,*_ = preflight(); write_json(args.receipt,r)
        print(json.dumps({k:v for k,v in r.items() if k not in ('patients','source_sha256')},indent=2))
    else: execute(args.receipt)


if __name__ == '__main__': main()
