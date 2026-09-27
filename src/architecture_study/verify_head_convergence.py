"""Verify saved diagnostic heads without fitting or loading patient shards."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import load_config, fingerprint
from .data import digest, write_json, aligned_values
from .analyze import metrics, paired_bootstrap
from .verify_controls import verify


def objective_gradient(x, labels, weights, fit, l2):
    """Independent analytic gradient in the standardized fitting coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(labels, dtype=float)
    w = np.asarray(weights, dtype=float); w = w/w.sum()
    mean = np.sum(x*w[:,None], axis=0)
    scale = np.maximum(np.sqrt(np.sum((x-mean)**2*w[:,None], axis=0)), 1e-6)
    z = (x-mean)/scale
    raw = np.asarray(fit['weight'], dtype=float).reshape(-1)
    theta = raw*scale
    bias = float(fit['bias'][0])+raw@mean
    logits = z@theta+bias
    p = torch.sigmoid(torch.tensor(logits)).numpy()
    residual = w*(p-y)
    gradient = np.r_[z.T@residual+2*l2*theta, residual.sum()]
    objective = np.sum(w*(np.logaddexp(0,logits)-y*logits))+l2*np.sum(theta**2)
    return float(objective), float(np.max(np.abs(gradient)))


def audit(run):
    old = verify(run)
    folder = run/'head-convergence-v1'
    marker = json.loads((folder/'complete.json').read_text())
    expected = {'protocol.json','comparison.json'} | {
        f'fold-{f}-{m}.json' for f in range(3) for m in ('R0','R1','process_only','R0_value_access')}
    if marker['status'] != 'complete' or not marker['all_stationarity_pass'] or set(marker['artifacts_sha256']) != expected:
        raise ValueError('Incomplete diagnostic')
    for name, checksum in marker['artifacts_sha256'].items():
        if digest(folder/name) != checksum:
            raise ValueError(f'Diagnostic checksum mismatch: {name}')
    c, _ = load_config(run/'config.toml')
    protocol = json.loads((folder/'protocol.json').read_text())
    for key, value in {'config_sha256':fingerprint(c), 'source_analysis_manifest':old['analysis_manifest_sha256'],
                       'code_sha256':digest(Path(__file__).with_name('head_convergence.py')),
                       'l2':c['optimization']['head_l2'], 'max_iter':500, 'tolerance_grad':1e-10,
                       'tolerance_change':1e-15, 'stationarity_gate':1e-8}.items():
        if protocol[key] != value:
            raise ValueError(f'Diagnostic protocol mismatch: {key}')
    comparison = json.loads((folder/'comparison.json').read_text())
    scores = {k:[] for k in ('R0','R1','process_only','R0_value_access','R0_random_frozen')}
    ys, ws, patients, rows, convergence = [], [], [], [], []
    for fold in range(3):
        base = run/f'fold-{fold}'
        refs = json.loads((base/'head_samples.json').read_text())
        held = json.loads((base/'assessment_samples.json').read_text())
        y = [int(r['episode_group'] != ['negative']) for r in refs]
        w = [1/r['inclusion_probability'] for r in refs]
        hy = [int(r['episode_group'] != ['negative']) for r in held]
        hw = [1/r['inclusion_probability'] for r in held]
        for name in ('R0','R1','process_only','R0_value_access'):
            d = json.loads((folder/f'fold-{fold}-{name}.json').read_text())
            recipe = 'R1' if name == 'R1' else 'R0'
            if d['fit_weights'] != w or d['fit_embeddings_sha256'] != digest(base/f'{recipe}-embeddings.npz'):
                raise ValueError('Diagnostic weights/embedding identity mismatch')
            with np.load(base/f'{recipe}-embeddings.npz', allow_pickle=False) as z:
                x, h = z['fit'], z['held']
                aligned_values([r['sample_id'] for r in refs], d['fit_sample_ids'], x)
                aligned_values(d['fit_sample_ids'], z['fit_ids'].tolist(), x)
                aligned_values([r['sample_id'] for r in held], d['assessment_sample_ids'], d['scores'])
                aligned_values(d['assessment_sample_ids'], z['held_ids'].tolist(), h)
            col = slice(None) if name in ('R0','R1') else (slice(32,None) if name == 'process_only' else list(range(32))+[57,58])
            fit = d['fit']
            objective, gradient = objective_gradient(x[:,col], y, w, fit, protocol['l2'])
            np.testing.assert_allclose(objective, fit['final_objective'], rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(gradient, fit['final_gradient_max'], rtol=1e-5, atol=1e-13)
            if gradient > protocol['stationarity_gate'] or not fit['stationarity_pass'] or objective > fit['initial_objective']+1e-12 or not 0 < fit['iterations'] <= 500:
                raise ValueError('Independent convergence check failed')
            replay = torch.sigmoid(torch.tensor(h[:,col],dtype=torch.float64) @ torch.tensor(fit['weight'],dtype=torch.float64).T+torch.tensor(fit['bias'],dtype=torch.float64)).flatten().numpy()
            np.testing.assert_array_equal(replay, d['scores'])
            row = {'fold':fold,'model':name,'stationarity_pass':True, **metrics(hy,d['scores'],hw)}
            if row not in comparison['rows']:
                raise ValueError('Diagnostic metric mismatch')
            rows.append(row); scores[name].extend(d['scores'])
            convergence.append({'fold':fold,'model':name,'gradient':gradient,'iterations':fit['iterations'],
                                'objective':objective, 'slope_norm':float(np.linalg.norm(fit['weight']))})
        random = json.loads((run/'analysis'/f'fold-{fold}-R0_random_frozen-predictions.json').read_text())
        scores['R0_random_frozen'].extend(random['scores'])
        ys.extend(hy); ws.extend(hw); patients.extend([r['patient'] for r in held])
    intervals = {}
    for left, right in [('process_only','R0'),('R0_value_access','R0'),('process_only','R0_value_access'),
                        ('R0_random_frozen','R0'),('R0','R1')]:
        result = paired_bootstrap(ys,scores[left],scores[right],ws,patients)
        result['difference'] = result.pop('difference_R1_minus_R0')
        intervals[f'{right}_minus_{left}'] = result
    return {'status':'verified', 'original_artifact_hashes_verified':old['hashed_artifacts'],
            'diagnostic_artifact_hashes_verified':len(expected), 'convergence':convergence,
            'rows':rows, 'pooled':{k:metrics(ys,v,ws) for k,v in scores.items()},
            'paired_intervals':intervals, 'fit_seconds':comparison['wall_seconds'],
            'scope':'saved training-fold artifacts only; no fitting or patient-shard access',
            'random_limitation':old['limitation'], 'complete_sha256':digest(folder/'complete.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Use a new output within the private study directory')
    write_json(args.output,audit(args.run))
    print('All diagnostic heads independently verified; original study unchanged.')
