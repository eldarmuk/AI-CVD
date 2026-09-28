"""Memory-bounded independent validation audit calculations."""
import numpy as np
from src.final_study.policy import calibrated


def fast_alerts(times,scores,events,threshold,task,cooldown_minutes,vital):
    times=np.asarray(times,dtype=np.int64); scores=np.asarray(scores); vital=np.asarray(vital,dtype=bool)
    if len(times)!=len(scores) or np.any(np.diff(times)<=0): raise ValueError('Unordered stream')
    grid=task.grid_minutes*60*10**9; horizon=task.horizon_minutes*60*10**9
    above=scores>=threshold
    crossing=above & np.r_[True,(~above[:-1]) | (np.diff(times)!=grid)]
    events=sorted((int(t),str(e)) for t,e in events)
    matched=set(); alerts=[]; last=None
    for i in np.flatnonzero(crossing):
        t=int(times[i])
        if last is not None and t-last<cooldown_minutes*60*10**9: continue
        last=t
        linked=next(((et,e) for et,e in events if t<et<=t+horizon and e not in matched),None)
        if linked: matched.add(linked[1])
        alerts.append({'prediction_ns':t,'episode_id':linked[1] if linked else None,
            'lead_minutes':(linked[0]-t)/60e9 if linked else None,'vital_evidence':bool(vital[i])})
    eligible=set(); by={True:set(),False:set()}
    for t,e in events:
        a,b=np.searchsorted(times,[t-horizon,t],side='left')
        if a<b:
            eligible.add(e)
            for v in (True,False):
                if np.any(vital[a:b]==v): by[v].add(e)
    return {'alerts':alerts,'eligible_episodes':sorted(eligible),'matched_episodes':sorted(matched),
        'eligible_by_evidence':{str(v):sorted(by[v]) for v in (True,False)},
        'cells_by_evidence':{str(v):int((vital==v).sum()) for v in (True,False)},
        'supported_cells':len(times),'span_cells':int((times[-1]-times[0])//grid+1) if len(times) else 0}


def accumulator():
    return {'patients':0,'event_patients':0,'episodes':0,'hits':0,'alerts':0,'cells':0,'span':0,'lead':[]}


def add(a,r):
    a['patients']+=1; a['event_patients']+=int(bool(r['eligible_episodes']))
    for target,key in [('episodes','eligible_episodes'),('hits','matched_episodes'),('alerts','alerts')]: a[target]+=len(r[key])
    a['cells']+=r['supported_cells']; a['span']+=r['span_cells']
    a['lead'].extend(x['lead_minutes'] for x in r['alerts'] if x['episode_id'] is not None)


def summary(a,task):
    days=a['cells']*task.grid_minutes/1440; false=a['alerts']-a['hits']
    return {'patients':a['patients'],'event_patients':a['event_patients'],'eligible_episodes':a['episodes'],
        'detected_episodes':a['hits'],'episode_sensitivity':a['hits']/a['episodes'] if a['episodes'] else None,
        'alerts':a['alerts'],'matched_alerts':a['hits'],'false_alerts':false,
        'alert_precision':a['hits']/a['alerts'] if a['alerts'] else None,
        'supported_patient_days':days,'false_alerts_per_supported_day':false/days if days else None,
        'supported_cells':a['cells'],'supported_fraction_of_within_patient_span':a['cells']/a['span'] if a['span'] else None,
        'lead_minutes_quantiles':np.quantile(a['lead'],[0,.25,.5,.75,1]).tolist() if a['lead'] else None}


def stream_metrics(y,p,intercept):
    """Exact unweighted tied-score ranking, sorted once; chunked probability sums."""
    y=np.asarray(y); p=np.asarray(p); n=len(y); positives=int(y.sum()); negatives=n-positives
    if not n: return {'samples':0,'reason':'empty'}
    order=np.argsort(-p,kind='stable'); sorted_p=p[order]; sorted_y=y[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_p)),n-1]
    cp=np.cumsum(sorted_y,dtype=np.float64)[ends]; cn=ends+1-cp
    dp=np.diff(np.r_[0,cp]); dn=np.diff(np.r_[0,cn])
    ap=float(np.sum(dp*(cp/(ends+1)))/positives) if positives else None
    auc=float(np.sum((cp-dp/2)*dn)/(positives*negatives)) if positives and negatives else None
    calibrated_sorted=calibrated(sorted_p,intercept)
    calibrated_ends=np.r_[np.flatnonzero(np.diff(calibrated_sorted)),n-1]
    same_ranking=np.array_equal(ends,calibrated_ends)
    cap,cauc=ap,auc
    if not same_ranking:
        ccp=np.cumsum(sorted_y,dtype=np.float64)[calibrated_ends]; ccn=calibrated_ends+1-ccp
        cdp=np.diff(np.r_[0,ccp]); cdn=np.diff(np.r_[0,ccn])
        cap=float(np.sum(cdp*(ccp/(calibrated_ends+1)))/positives) if positives else None
        cauc=float(np.sum((ccp-cdp/2)*cdn)/(positives*negatives)) if positives and negatives else None
    del calibrated_sorted,calibrated_ends
    del order,sorted_p,sorted_y,ends,cp,cn,dp,dn
    diagnostics={}
    for name,b in [('raw',None),('calibrated',intercept)]:
        total=ll=brier=0.
        for start in range(0,n,1000000):
            target=y[start:start+1000000].astype(float); scores=p[start:start+1000000]
            if b is not None: scores=calibrated(scores,b)
            bounded=np.clip(scores,1e-12,1-1e-12)
            total+=float(scores.sum()); brier+=float(np.sum((scores-target)**2))
            ll+=float(np.sum(-target*np.log(bounded)-(1-target)*np.log1p(-bounded)))
        diagnostics[name]={'mean_risk':total/n,'log_loss':ll/n,'brier':brier/n}
    return {'samples':n,'positive_windows':positives,'prevalence':positives/n,'ap':ap,'auroc':auc,
            'calibrated_ap':cap,'calibrated_auroc':cauc,'calibration_ranking_unchanged':bool(same_ranking),
            'ap_over_prevalence':ap/(positives/n) if positives else None,'probabilities':diagnostics,
            'ranking_note':'raw ranking; intercept preserves ranking where numerical clipping/saturation does not create ties'}
