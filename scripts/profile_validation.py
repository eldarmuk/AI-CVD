"""Bounded validation subset benchmark, never a full dataset or test pass."""
import argparse
import time
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from scripts.validation_support import metadata,SCORES,POLICY,read,vector_windows
from src.architecture_study.data import digest,write_json
from src.architecture_study.projection import project
from src.final_study.policy import calibrated,patient_alerts


def profile():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    spec,c,task,model,scaler,root,manifest,export,cal,pol=metadata()
    cases=[]; load_seconds=0.
    # Fixed without looking at outcomes: first 3 manifest patients, first 256 rows each.
    for r,s in zip(manifest[:3],export['shards'][:3]):
        begun=time.perf_counter(); path=root/s['file']
        if digest(path)!=s['sha256'] or digest(SCORES/r['file'])!=r['sha256']: raise ValueError('Subset hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            x=z['features']; ends=z['end_rows'][:256]; start=int(z['grid_start_ns'])
            native_ids=z['sample_ids'][:256]
        with np.load(SCORES/r['file'],allow_pickle=False) as z:
            original=z['scores'][:len(ends)]; times=z['times'][:len(ends)]; vital=z['vital'][:len(ends)]
            ids=z['sample_ids'][:len(ends)].tolist()
        if ids!=['sample_'+bytes(v).hex() for v in native_ids]: raise ValueError('Subset ID alignment failure')
        np.testing.assert_array_equal(times,start+(ends.astype(np.int64)+1)*task.grid_minutes*60*10**9)
        load_seconds+=time.perf_counter()-begun
        cases.append((x,ends,original,times,vital,r['events'],ids))
    intercept=read(POLICY/'calibration.json')['intercept']; threshold=read(POLICY/'threshold.json')['selected']['threshold']
    measurements=[]
    options=[('reference',4,64,'cpu'),('vector',4,64,'cpu'),('vector',1,64,'cpu'),('vector',4,256,'cpu')]
    if torch.cuda.is_available(): options.append(('vector',4,256,'cuda'))
    for mode,threads,batch,device in options:
        torch.set_num_threads(threads); model=model.to(device).eval()
        prep=forward=0.; outputs=[]; errors=[]; exact=True; policies=True; repeated=True
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        for x,ends,original,times,vital,events,ids in cases:
            out=[]
            for start in range(0,len(ends),batch):
                selected=ends[start:start+batch]; begun=time.perf_counter()
                windows=np.stack([x[e-task.sequence_steps+1:e+1] for e in selected]) if mode=='reference' else vector_windows(x,selected,task.sequence_steps)
                tensor=torch.tensor(windows,device=device)
                view=project(tensor,scaler['mean'],scaler['scale'],task)
                if device=='cuda': torch.cuda.synchronize()
                prep+=time.perf_counter()-begun; begun=time.perf_counter()
                with torch.no_grad(): pred=model(view).sigmoid().cpu().numpy()
                forward+=time.perf_counter()-begun
                with torch.no_grad(): repeated &= np.array_equal(pred,model(view).sigmoid().cpu().numpy())
                out.extend(pred.tolist())
            out=np.array(out); exact &= np.array_equal(out,original)
            errors.append(float(np.max(np.abs(out-original))))
            # Tolerances fixed before timing; policy equivalence is separately mandatory.
            outputs.append(bool(np.allclose(out,original,rtol=1e-5,atol=1e-10)))
            cooldown=spec['validation']['cooldown_minutes']
            policies &= patient_alerts(times,calibrated(out,intercept),events,threshold,task,cooldown,vital)==patient_alerts(times,calibrated(original,intercept),events,threshold,task,cooldown,vital)
        n=sum(len(case[1]) for case in cases)
        measurements.append({'mode':mode,'threads':threads,'batch':batch,'device':device,'samples':n,
            'preparation_seconds':prep,'forward_seconds':forward,'examples_per_second':n/(prep+forward),
            'exact_saved_scores':bool(exact),'within_tolerance':all(outputs),'deterministic':bool(repeated),
            'identical_subset_alert_policy':bool(policies),'max_absolute_error':max(errors),
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None})
    return {'scope':'first 3 validation patients, first 256 windows each, no outcome-based selection',
            'patient_ids':[r['patient'] for r in manifest[:3]],'sample_ids':[case[-1] for case in cases],
            'source_load_and_hash_seconds':load_seconds,'tolerance':{'rtol':1e-5,'atol':1e-10},
            'measurements':measurements,'limitation':'small subset/cold-start timing, no full-stream equivalence claim; no test access'}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise ValueError('Use new private output')
    result=profile(); write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('patient_ids','sample_ids')},indent=2))
