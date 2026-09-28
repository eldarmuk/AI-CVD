"""Manual exact patient-cluster bootstrap of sealed saved test predictions only."""
import argparse
import time
import numpy as np
from scipy.stats import beta
from scripts.test_evaluation import OUTPUT, read, code_hashes
from src.architecture_study.data import write_json, digest
from src.final_study.config import verify_seal, seal, load

REPLICATES = 200
SEED = 101027


def weighted_ranking(sorted_y, sorted_patients, tie_ends, multiplicity):
    """Whole patients get one integer bootstrap multiplicity; ties pooled exactly."""
    w = multiplicity[sorted_patients]
    cp = np.cumsum(w*sorted_y,dtype=np.float64)[tie_ends]
    cn = np.cumsum(w*(1-sorted_y),dtype=np.float64)[tie_ends]
    dp = np.diff(np.r_[0.,cp]); dn = np.diff(np.r_[0.,cn])
    precision = np.divide(cp,cp+cn,out=np.zeros_like(cp),where=(cp+cn)>0)
    ap = float(np.sum(dp*precision)/cp[-1]) if cp[-1] else None
    auc = float(np.sum((cp-dp/2)*dn)/(cp[-1]*cn[-1])) if cp[-1] and cn[-1] else None
    return auc,ap


def binomial_interval(hits, episodes):
    if not episodes: return None
    return [float(beta.ppf(.025,hits,episodes-hits+1)) if hits else 0.,
            float(beta.ppf(.975,hits+1,episodes-hits)) if hits<episodes else 1.]


def interval(values):
    valid = [v for v in values if v is not None]
    return {'percentile_95':np.quantile(valid,[.025,.975]).tolist() if valid else None,
            'valid_replicates':len(valid),'undefined_replicates':len(values)-len(valid)}


def run():
    output = OUTPUT/'uncertainty-v1'
    if output.exists(): raise FileExistsError('Immutable uncertainty output exists; explicit recovery review required')
    verify_seal(OUTPUT)
    manifest = read(OUTPUT/'final_result_manifest.json')
    if manifest['source_sha256'] != code_hashes(): raise ValueError('Evaluation source changed')
    patients = read(OUTPUT/'patients.json'); _,_,task = load()
    output.mkdir(); begun = time.perf_counter()
    write_json(output/'started.json',{'seed':SEED,'replicates':REPLICATES,'unit':'patient',
        'test_complete_sha256':digest(OUTPUT/'complete.json'),'source_sha256':code_hashes()})
    y = np.load(OUTPUT/'labels.npy',mmap_mode='r'); scores = np.load(OUTPUT/'scores.npy',mmap_mode='r')
    pid = np.load(OUTPUT/'patient_index.npy',mmap_mode='r')
    order = np.argsort(-scores,kind='stable')
    sorted_scores = scores[order]; sy = y[order]; sp = pid[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)),len(y)-1]
    del order,sorted_scores
    episodes = np.array([len(p['operational']['eligible_episodes']) for p in patients])
    hits = np.array([len(p['operational']['matched_episodes']) for p in patients])
    false = np.array([len(p['operational']['alerts']) for p in patients])-hits
    days = np.array([p['samples'] for p in patients])*task.grid_minutes/1440
    rng = np.random.default_rng(SEED); rows = []; draws = []
    for i in range(REPLICATES):
        selected = rng.integers(0,len(patients),len(patients)); draws.append(selected)
        w = np.bincount(selected,minlength=len(patients))
        auc,ap = weighted_ranking(sy,sp,ends,w)
        ne = int(w@episodes)
        row = {'replicate':i,'auroc':auc,'ap':ap,'episode_sensitivity':float(w@hits/ne) if ne else None,
            'unmatched_alerts_per_supported_day':float(w@false/(w@days))}
        rows.append(row); write_json(output/f'replicate-{i:04d}.json',row)
        print(f'Patient bootstrap {i+1}/{REPLICATES}',flush=True)
    np.save(output/'patient_draw_indices.npy',np.asarray(draws))
    result = {'seed':SEED,'replicates':REPLICATES,'method':'ordinary patient-cluster percentile bootstrap, exact weighted tied-score ranking',
        'intervals':{key:interval([r[key] for r in rows]) for key in rows[0] if key!='replicate'},
        'episode_binomial_95_supplementary':binomial_interval(int(hits.sum()),int(episodes.sum())),
        'limitations':'200 replicates give coarse tail precision. Sparse event patients may produce undefined ranking/sensitivity replicates, explicitly counted. Patients are resampled with every window and episode together. Binomial interval assumes independent episodes and is supplementary because repeated episodes within a patient can correlate. No uncertainty calculation refits calibration, model or policy.',
        'seconds':time.perf_counter()-begun}
    write_json(output/'uncertainty_results.json',result)
    seal(output,purpose='fixed patient-cluster uncertainty on sealed test results')
    print('TEST UNCERTAINTY COMPLETE: uncertainty-v1/complete.json',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge-long-analysis',action='store_true',required=True)
    parser.parse_args(); run()
