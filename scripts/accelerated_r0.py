"""Bounded CPU inference adapter. No dataset/test runner and no fitting."""
import numpy as np
import torch
from scripts.validation_support import vector_windows
from src.architecture_study.projection import project

ATOL=1e-10
RTOL=1e-5
BATCH=256
REFERENCE_BATCH=64


def check_equivalence(candidate,reference):
    if candidate.shape!=reference.shape or not np.isfinite(candidate).all() or not np.allclose(candidate,reference,rtol=RTOL,atol=ATOL):
        raise ValueError('Numerical equivalence failed; do not relax tolerances or continue evaluation')


def score_patient(model,features,ends,scaler,task,threshold,intercept):
    """Ordered eligible rows; reference guards never use outcomes.

    Each patient has first/last original-size batch canaries. Every original-size
    batch with a candidate near the operating threshold is replayed on CPU64,
    checked against tolerance, and replaced by exact reference scores. This guards
    threshold decisions; whole-stream numerical equivalence is not asserted.
    """
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    if next(model.parameters()).device.type!='cpu': raise ValueError('Only reviewed CPU adapter supported')
    model.eval(); ends=np.asarray(ends,dtype=np.int64)
    if np.any(np.diff(ends)<=0): raise ValueError('Eligible sequence order changed')
    n=len(ends); result=np.empty(n,dtype=np.float64)
    def infer(a,b,batch):
        pieces=[]
        for start in range(a,b,batch):
            x=vector_windows(features,ends[start:min(start+batch,b)],task.sequence_steps)
            view=project(torch.tensor(x),scaler['mean'],scaler['scale'],task)
            with torch.no_grad(): pieces.append(model(view).sigmoid().numpy())
        return np.concatenate(pieces) if pieces else np.empty(0,dtype=np.float32)
    result[:]=infer(0,n,BATCH)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite inference')
    # No-alert threshold above one is a legitimate frozen operating point.
    guard=set()
    if n: guard.update([0,((n-1)//REFERENCE_BATCH)*REFERENCE_BATCH])
    if 0<threshold<1:
        raw=1/(1+np.exp(-(np.log(threshold)-np.log1p(-threshold)-intercept)))
        near=np.flatnonzero(np.abs(result-raw)<=2*(ATOL+RTOL*abs(raw)))
        guard.update((near//REFERENCE_BATCH*REFERENCE_BATCH).tolist())
    else: near=[]
    for start in sorted(guard):
        stop=min(start+REFERENCE_BATCH,n); ref=infer(start,stop,REFERENCE_BATCH)
        check_equivalence(result[start:stop],ref)
        # Verify repeated reference inference determinism and preserve exact values.
        if not np.array_equal(ref,infer(start,stop,REFERENCE_BATCH)): raise ValueError('Nondeterministic reference inference')
        result[start:stop]=ref
    return result,{'guarded_reference_batches':len(guard),'near_threshold_windows':len(near),
                   'samples':n,'batch':BATCH,'reference_batch':REFERENCE_BATCH,'atol':ATOL,'rtol':RTOL}
