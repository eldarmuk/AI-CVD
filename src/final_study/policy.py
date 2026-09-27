"""Validation-only development and causal first-crossing episode metrics."""
import numpy as np
from src.ai_cvd.dataset import patient_split
from src.architecture_study.config import fingerprint


def authorize_validation(split,patients,task):
    if split != 'validation' or any(patient_split(p,task) != 'validation' for p in patients):
        raise ValueError('Calibration/policy development requires canonical validation patients')


def calibration_partition(patient,seed):
    return int(fingerprint([seed,patient]),16) % 2 == 0


def calibrated(scores,intercept):
    p = np.clip(np.asarray(scores,dtype=float),1e-12,1-1e-12)
    logits = np.log(p)-np.log1p(-p)+intercept
    return np.exp(-np.logaddexp(0,-logits))


def fit_intercept(scores,labels,*,split,patients,task):
    authorize_validation(split,patients,task)
    p,y = np.asarray(scores,dtype=float),np.asarray(labels)
    if len(p) != len(y) or len(p) != len(patients) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid natural-stream calibration data')
    if not len(y) or len(np.unique(y)) < 2:
        return {'intercept':0.,'status':'identity_insufficient_classes','samples':len(y),'positives':int(y.sum())}
    lo,hi = -40.,40.
    for _ in range(100):
        middle = (lo+hi)/2
        if calibrated(p,middle).sum() > y.sum():
            hi = middle
        else:
            lo = middle
    return {'intercept':(lo+hi)/2,'status':'fitted','samples':len(y),'positives':int(y.sum()),
            'method':'unweighted natural-stream intercept only','boundary_hit':abs((lo+hi)/2)>39.9}


def patient_alerts(times,scores,events,threshold,task,cooldown_minutes,vital=None):
    times = np.asarray(times,dtype=np.int64); scores = np.asarray(scores,dtype=float)
    if len(times) != len(scores) or np.any(np.diff(times)<=0) or not np.isfinite(scores).all():
        raise ValueError('Unordered/misaligned stream')
    events = sorted((int(t),str(e)) for t,e in events)
    if len({e for _,e in events}) != len(events):
        raise ValueError('Duplicate clinical episode')
    horizon = task.horizon_minutes*60*10**9; grid = task.grid_minutes*60*10**9
    cooldown = cooldown_minutes*60*10**9
    vital = np.ones(len(times),dtype=bool) if vital is None else np.asarray(vital,dtype=bool)
    if vital.shape != times.shape:
        raise ValueError('Evidence alignment mismatch')
    eligible = set(); eligibility = {True:set(),False:set()}
    for t,e in events:
        left,right = np.searchsorted(times,[t-horizon,t],side='left')
        if left < right:
            eligible.add(e)
            for v in (True,False):
                if np.any(vital[left:right] == v):
                    eligibility[v].add(e)
    matched = set(); alerts = []; previous_above = False; last = None
    for i,(t,p) in enumerate(zip(times,scores)):
        contiguous = i>0 and t-times[i-1] == grid
        above = p >= threshold
        crossing = above and (not contiguous or not previous_above)
        previous_above = above
        if not crossing or (last is not None and t-last < cooldown):
            continue
        last = t
        linked = next(((et,e) for et,e in events if t < et <= t+horizon and e not in matched),None)
        if linked:
            matched.add(linked[1])
        alerts.append({'prediction_ns':int(t),'episode_id':linked[1] if linked else None,
                       'lead_minutes':(linked[0]-int(t))/60e9 if linked else None,'vital_evidence':bool(vital[i])})
    return {'alerts':alerts,'eligible_episodes':sorted(eligible),'matched_episodes':sorted(matched),
            'eligible_by_evidence':{str(v):sorted(eligibility[v]) for v in (True,False)},
            'cells_by_evidence':{str(v):int((vital==v).sum()) for v in (True,False)},
            'supported_cells':len(times),
            'span_cells':int((times[-1]-times[0])//grid+1) if len(times) else 0}


def evaluate(stream,threshold,task,cooldown_minutes,intercept=0.):
    results = {p:patient_alerts(d['times'],calibrated(d['scores'],intercept),d['events'],threshold,
                             task,cooldown_minutes,d['vital']) for p,d in stream.items()}
    n = sum(len(r['eligible_episodes']) for r in results.values())
    hits = sum(len(r['matched_episodes']) for r in results.values())
    alerts = [a for r in results.values() for a in r['alerts']]
    cells = sum(r['supported_cells'] for r in results.values())
    span = sum(r['span_cells'] for r in results.values())
    days = cells*task.grid_minutes/1440
    false = sum(a['episode_id'] is None for a in alerts)
    leads = [a['lead_minutes'] for a in alerts if a['episode_id'] is not None]
    strata = {}
    for v in (True,False):
        selected = [a for a in alerts if a['vital_evidence'] == v]
        denom = sum(len(r['eligible_by_evidence'][str(v)]) for r in results.values())
        sd = sum(r['cells_by_evidence'][str(v)] for r in results.values())*task.grid_minutes/1440
        sh = sum(a['episode_id'] is not None for a in selected)
        strata[str(v)] = {'alerts':len(selected),'matched_alerts':sh,'eligible_episodes':denom,
            'episode_sensitivity':sh/denom if denom else None,'alert_precision':sh/len(selected) if selected else None,
            'false_alerts_per_supported_day':(len(selected)-sh)/sd if sd else None}
    return {'eligible_episodes':n,'detected_episodes':hits,'episode_sensitivity':hits/n if n else None,
            'alerts':len(alerts),'alert_precision':hits/len(alerts) if alerts else None,
            'false_alerts':false,'supported_patient_days':days,'false_alerts_per_supported_day':false/days if days else None,
            'lead_minutes_quantiles':np.quantile(leads,[0,.25,.5,.75,1]).tolist() if leads else None,
            'supported_cells':cells,'supported_fraction_of_within_patient_span':cells/span if span else None,
            'evidence_strata':strata,'patients':results}


def select_threshold(stream,intercept,*,split,task,config):
    authorize_validation(split,stream,task)
    if not stream:
        raise ValueError('No policy validation patients')
    pooled = np.concatenate([calibrated(d['scores'],intercept) for d in stream.values()])
    if not len(pooled):
        raise ValueError('Empty policy stream')
    # Prespecified high-tail resolution is necessary for rare outcomes/low alert budgets.
    quantiles = np.unique(np.r_[np.linspace(0,.9,10),1-np.geomspace(.1,1e-6,101),1.])
    candidates = np.unique(np.r_[np.quantile(pooled,quantiles),1.000000000001])
    best = None; best_key = None; table = []
    for threshold in candidates:
        result = evaluate(stream,float(threshold),task,config['cooldown_minutes'],intercept)
        if not result['eligible_episodes']:
            raise ValueError('No eligible policy events; cannot develop operating point')
        row = {k:v for k,v in result.items() if k != 'patients'}; row['threshold'] = float(threshold)
        table.append(row)
        if result['false_alerts_per_supported_day'] <= config['false_alerts_per_supported_day_cap']:
            key = (result['episode_sensitivity'],result['alert_precision'] or 0.,float(threshold))
            if best_key is None or key > best_key:
                best_key = key; best = row
    return {'selected':best,'candidates':table,'criterion':config['threshold_objective']}
