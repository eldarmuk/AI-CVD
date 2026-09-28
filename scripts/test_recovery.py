"""Explicit reviewed continuation; audit never scores or reports performance."""
import argparse
import hashlib
import json
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scripts import test_evaluation as ev

ORIGINAL_COMMIT = '7794c5ddf92848c1eb379b4c4f1456090d544d35'
TYPES = {'scores':'float64','labels':'uint8','vital':'bool','primitive':'bool','patient_index':'int32'}


def export_prefix(path, count):
    """Decode only completed patient records, not remaining event metadata."""
    text = path.read_text(encoding='utf-8'); decoder = json.JSONDecoder()
    cursor = text.index('[',text.index('"shards"'))+1
    rows = []
    for _ in range(count):
        while text[cursor].isspace() or text[cursor]==',': cursor += 1
        row,cursor = decoder.raw_decode(text,cursor); rows.append(row)
    return rows


def verify_prefix(output, root, blind, task, e, windows):
    """Read-only integrity verification. No forward pass and no aggregate metrics."""
    records = sorted(output.glob('patient-*.json')); chunks = sorted(output.glob('patient-*.npz'))
    if len(records)!=len(chunks) or not records: raise ValueError('Partial/unpaired or empty patient prefix: explicit recovery review required')
    if any((output/name).exists() for name in ('complete.json','patients.json','test_results.json','final_result_manifest.json')):
        raise ValueError('Completed/aggregate-stage artifacts exist; this continuation is not authorized')
    allowed = {'started.json'} | {f'{k}.npy' for k in TYPES} | {p.name for p in records+chunks}
    for i,p in enumerate(sorted(output.glob('recovery-*.json')),1):
        if p.name!=f'recovery-{i:04d}.json': raise ValueError('Recovery provenance sequence changed')
        allowed.add(p.name)
    if {p.name for p in output.iterdir()} != allowed: raise ValueError('Unexpected partial artifact; preserve for explicit review')
    source = export_prefix(root/'exports/test-stream.json',len(records))
    pooled = {k:np.load(output/f'{k}.npy',mmap_mode='r') for k in TYPES}
    for k,a in pooled.items():
        if a.shape!=(windows,) or a.dtype!=np.dtype(TYPES[k]): raise ValueError('Pooled array schema changed')
    pos = 0; rows = []
    for i,(path,s) in enumerate(zip(records,source)):
        if path.name!=f'patient-{i:06d}.json' or chunks[i].name!=f'patient-{i:06d}.npz':
            raise ValueError('Noncontiguous patient prefix')
        if i>=len(blind['shards']) or any(s[k]!=v for k,v in blind['shards'][i].items()):
            raise ValueError('Source population mismatch')
        r = ev.read(path); ev.check_hash(chunks[i],r['sha256']); ev.check_hash(root/s['file'],s['sha256'])
        expected = {'patient':s['senior_id'],'file':chunks[i].name,'source_shard_sha256':s['sha256'],
                    'events':s['events'],'samples':s['samples'],'offset':pos}
        if any(r[k]!=v for k,v in expected.items()): raise ValueError('Patient manifest identity changed')
        with np.load(root/s['file'],allow_pickle=False) as z:
            a = {k:z[k] for k in ('features','end_rows','grid_start_ns','sample_ids','target','first_event','event_stop')}
        ends,times,ids = ev.verify_rows(r['patient'],a,s['events'],s['samples'],task)
        with np.load(chunks[i],allow_pickle=False) as z: d = {k:z[k] for k in z.files}
        vital,primitive = ev.evidence(a['features'],ends,task)
        expected = {'sample_ids':ids,'times':times,'labels':a['target'],'first_event':a['first_event'],
                    'event_stop':a['event_stop'],'vital':vital,'primitive':primitive}
        if set(d)!=set(expected)|{'scores','calibrated_scores'}: raise ValueError('Prediction schema changed')
        for k,v in expected.items(): np.testing.assert_array_equal(d[k],v)
        if d['scores'].shape!=times.shape or not np.isfinite(d['scores']).all() or np.any((d['scores']<0)|(d['scores']>1)):
            raise ValueError('Invalid saved risk')
        np.testing.assert_array_equal(d['calibrated_scores'],ev.calibrated(d['scores'],e['calibration']['intercept']))
        result = ev.fast_alerts(times,d['calibrated_scores'],s['events'],e['threshold'],task,e['alert_policy']['cooldown_minutes'],vital)
        if r['operational']!=result: raise ValueError('Saved policy record changed')
        for name,mask in [('physiology_observed',vital),('no_physiology',~vital),('no_primitive',~primitive)]:
            if r['strata'][name]!=ev.subgroup_result(result,times,mask,s['events'],task): raise ValueError('Evidence attribution changed')
        for k in pooled:
            expected = np.full(len(times),i) if k=='patient_index' else d[k]
            np.testing.assert_array_equal(pooled[k][pos:pos+len(times)],expected)
        guards = r['guards']
        if any(guards[k]!=v for k,v in {'samples':len(times),'batch':256,'reference_batch':64,'atol':1e-10,'rtol':1e-5}.items()):
            raise ValueError('Scoring guard configuration changed')
        rows.append(r); pos += len(times)
        if (i+1)%100==0: print(f'Integrity verified {i+1}/{len(records)} completed patients; no performance aggregation',flush=True)
    # Cache tails must be the untouched zero-initialized allocation. Never repair silently.
    for a in pooled.values():
        for start in range(pos,windows,1000000):
            if np.any(a[start:start+1000000]!=0): raise ValueError('Nonempty uncommitted cache tail')
    del pooled
    snapshot = {p.name:ev.digest(p) for p in sorted(output.iterdir())}
    return rows,pos,snapshot


def audit(receipt_path):
    begun = time.perf_counter()
    bundle = ev.preflight(); current,e,task,model,scaler,root,blind = bundle
    original = ev.read(receipt_path)
    started = ev.read(ev.OUTPUT/'started.json')
    if started['preflight']!=original or original['git_commit']!=ORIGINAL_COMMIT:
        raise ValueError('Original execution receipt changed')
    if {k:v for k,v in current.items() if k not in ('git_commit','source_sha256')} != {k:v for k,v in original.items() if k not in ('git_commit','source_sha256')}:
        raise ValueError('Scientific/population preflight changed')
    # Only the explicit continuation amendment may differ from original runner code.
    for name,h in original['source_sha256'].items():
        if name=='scripts/test_evaluation.py':
            blob = subprocess.check_output(['git','show',f'{ORIGINAL_COMMIT}:{name}'],cwd=ev.ROOT)
            if hashlib.sha256(blob).hexdigest()!=h: raise ValueError('Original runner source cannot be established')
        else: ev.check_hash(ev.ROOT/name,h)
    if set(current['source_sha256']) != set(original['source_sha256'])|{'scripts/test_recovery.py'}:
        raise ValueError('Unreviewed source set')
    for path in sorted(ev.OUTPUT.glob('recovery-*.json')):
        previous = ev.read(path)
        if previous['original_preflight']!=original or previous['audit']['evaluation_spec_sha256']!=ev.SPEC_HASH or previous['audit']['engineering_sha256']!=ev.ENGINEERING_HASH:
            raise ValueError('Previous recovery bindings changed')
        for name,h in previous['audit']['snapshot_sha256'].items():
            if Path(name).name!=name: raise ValueError('Invalid recovery artifact path')
            if name not in {f'{k}.npy' for k in TYPES}: ev.check_hash(ev.OUTPUT/name,h)
    rows,pos,snapshot = verify_prefix(ev.OUTPUT,root,blind,task,e,current['windows'])
    report = {'status':'passed_read_only_recovery_audit','completed_patients':len(rows),'completed_windows':pos,
        'next_patient_ordinal_one_based':len(rows)+1,'remaining_patients':current['patient_count']-len(rows),
        'remaining_windows':current['windows']-pos,'original_receipt_sha256':ev.digest(receipt_path),
        'original_started_sha256':ev.digest(ev.OUTPUT/'started.json'),'snapshot_sha256':snapshot,
        'original_console_sha256':ev.digest(ev.BASE/'r0-final-v1-test.console.log'),
        'evaluation_spec_sha256':ev.SPEC_HASH,'engineering_sha256':ev.ENGINEERING_HASH,
        'recovery_source_sha256':current['source_sha256'],'recovery_git_commit':current['git_commit'],
        'aggregate_metrics_computed':False,'complete_marker_present':False,'pooled_prefix_exact':True,
        'seconds':time.perf_counter()-begun,
        'limits':'Saved scores authenticated by original patient checksums; no model rescoring. Patient policy records replayed for integrity only; no performance interpretation.'}
    return report,bundle,original,rows


def prepare(receipt_path,audit_path):
    approved = ev.read(audit_path)
    report,bundle,original,rows = audit(receipt_path)
    if {k:v for k,v in approved.items() if k!='seconds'} != {k:v for k,v in report.items() if k!='seconds'}:
        raise ValueError('Recovery audit snapshot/code stale; no writes performed')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ev.ROOT,text=True).strip():
        raise ValueError('Commit recovery amendment before continuation')
    sequence = len(list(ev.OUTPUT.glob('recovery-*.json')))+1
    ev.write_json(ev.OUTPUT/f'recovery-{sequence:04d}.json',{
        'reason':'Accidental process interruption; explicit integrity-verified continuation',
        'user_attestation':'No test-derived scientific decision or tuning between attempts',
        'utc':datetime.now(timezone.utc).isoformat(),'audit_sha256':ev.digest(audit_path),'audit':report,
        'original_preflight':original,'current_preflight':bundle[0],
        'reuse':'Completed patient NPZ/JSON and pooled prefixes remain byte-for-byte unchanged',
        'runtime_note':'Final seconds/resources measure this continuation only; original start and prior progress are preserved separately'})
    return bundle,original,rows


@contextmanager
def exclusive_recovery():
    """Windows kernel mutex: released by OS on exit/crash; no stale file removal."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None,True,'Local\\AI_CVD_R0_FINAL_TEST_RECOVERY')
    error = ctypes.get_last_error()
    if not handle: raise OSError(error,'Cannot reserve recovery mutex')
    if error==183:
        kernel.CloseHandle(handle); raise RuntimeError('A recovery process already holds the mutex')
    try: yield
    finally:
        kernel.ReleaseMutex(handle); kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for mode in ('audit','continue'):
        p=sub.add_parser(mode); p.add_argument('--receipt',type=Path,required=True); p.add_argument('--audit',type=Path,required=True)
        if mode=='continue': p.add_argument('--attest-no-test-driven-changes',action='store_true',required=True)
    args=parser.parse_args()
    if not args.audit.resolve().is_relative_to(ev.BASE.resolve()) or args.audit.resolve().is_relative_to(ev.OUTPUT.resolve()):
        raise ValueError('Use a new private audit file outside the interrupted directory')
    with exclusive_recovery():
        if args.command=='audit':
            if args.audit.exists(): raise FileExistsError('Audit receipt already exists')
            report,*_=audit(args.receipt); ev.write_json(args.audit,report)
            print(json.dumps({k:v for k,v in report.items() if k not in ('snapshot_sha256','recovery_source_sha256')},indent=2))
        else: ev.execute(args.receipt,recovery_audit=args.audit)


if __name__=='__main__': main()
