"""Separate numerical sensitivity check on saved embeddings; never updates encoders.

Not the frozen study. Same weighted logistic objective and L2, tighter stopping
tolerances and a 500-iteration cap. All four saved representations are checked;
no best-of-original/revised selection and no canonical patient-shard reads.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import digest, write_json, aligned_values
from .config import load_config, fingerprint
from .verify_controls import verify
from .analyze import metrics


def solve(x, y, weights, l2):
    x, y, w = [torch.as_tensor(a, dtype=torch.float64) for a in (x,y,weights)]
    w = w/w.sum()
    mean = (x*w[:,None]).sum(0)
    scale = ((x-mean).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-6)
    z = (x-mean)/scale
    head = torch.nn.Linear(x.shape[1],1,dtype=torch.float64)
    with torch.no_grad():
        head.weight.zero_(); head.bias.fill_(torch.logit((y*w).sum().clamp(1e-9,1-1e-9)))
    opt = torch.optim.LBFGS(head.parameters(), max_iter=500, tolerance_grad=1e-10,
                            tolerance_change=1e-15, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(head(z).squeeze(-1),y,reduction='none')*w).sum()+l2*head.weight.square().sum()
        loss.backward()
        return loss
    initial = float(closure().detach())
    initial_gradient = torch.cat([p.grad.flatten() for p in head.parameters()])
    opt.step(closure)
    final = float(closure().detach())
    gradient = torch.cat([p.grad.flatten() for p in head.parameters()])
    if not torch.isfinite(gradient).all() or not np.isfinite(final) or final > initial+1e-12:
        raise ValueError('Numerical solver failed')
    weight = head.weight.detach()/scale
    bias = head.bias.detach()-(weight*mean).sum()
    return {'weight':weight.tolist(), 'bias':bias.tolist(), 'initial_objective':initial,
            'final_objective':final, 'initial_gradient_max':float(initial_gradient.abs().max()),
            'initial_directional_derivative':float(-initial_gradient.square().sum()),
            'final_gradient_max':float(gradient.abs().max()),
            'stationarity_pass':bool(gradient.abs().max() <= 1e-8),
            'iterations':opt.state[head.weight]['n_iter']}


def run(source, output):
    if output.exists() or not output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Use a new private subdirectory; partial runs are not overwritten')
    torch.set_num_threads(4)
    audit = verify(source)
    c, _ = load_config(source/'config.toml')
    output.mkdir()
    write_json(output/'protocol.json', {'purpose':'numerical convergence sensitivity, not architecture selection',
        'config_sha256':fingerprint(c), 'source_analysis_manifest':audit['analysis_manifest_sha256'],
        'code_sha256':digest(Path(__file__)), 'l2':c['optimization']['head_l2'],
        'max_iter':500, 'tolerance_grad':1e-10, 'tolerance_change':1e-15,
        'stationarity_gate':1e-8, 'random_control':'not refit: embeddings unavailable',
        'resume':'no; refuses existing directory, preserves partial artifacts'})
    rows = []; start = time.perf_counter()
    for fold in range(3):
        folder = source/f'fold-{fold}'
        fit = json.loads((folder/'head_samples.json').read_text())
        held = json.loads((folder/'assessment_samples.json').read_text())
        y = [int(r['episode_group'] != ['negative']) for r in fit]
        w = [1/r['inclusion_probability'] for r in fit]
        hy = [int(r['episode_group'] != ['negative']) for r in held]
        hw = [1/r['inclusion_probability'] for r in held]
        for name in ('R0','R1','process_only','R0_value_access'):
            recipe = 'R1' if name == 'R1' else 'R0'
            with np.load(folder/f'{recipe}-embeddings.npz', allow_pickle=False) as z:
                x, h = z['fit'], z['held']
                aligned_values([r['sample_id'] for r in fit], z['fit_ids'].tolist(), x)
                aligned_values([r['sample_id'] for r in held], z['held_ids'].tolist(), h)
            columns = slice(None) if name in ('R0','R1') else (slice(32,None) if name == 'process_only' else list(range(32))+[57,58])
            fitted = solve(x[:,columns],y,w,c['optimization']['head_l2'])
            scores = torch.sigmoid(torch.as_tensor(h[:,columns],dtype=torch.float64) @ torch.tensor(fitted['weight'],dtype=torch.float64).T + torch.tensor(fitted['bias'],dtype=torch.float64)).flatten().tolist()
            write_json(output/f'fold-{fold}-{name}.json', {'fit_sample_ids':[r['sample_id'] for r in fit],
                'fit_weights':w, 'assessment_sample_ids':[r['sample_id'] for r in held],
                'scores':scores,'fit':fitted,'fit_embeddings_sha256':digest(folder/f'{recipe}-embeddings.npz')})
            rows.append({'fold':fold,'model':name, 'stationarity_pass':fitted['stationarity_pass'], **metrics(hy,scores,hw)})
            print(f'Completed numerical check fold {fold}/{name}; stationary={fitted["stationarity_pass"]}',flush=True)
    write_json(output/'comparison.json', {'rows':rows, 'wall_seconds':time.perf_counter()-start})
    write_json(output/'complete.json', {'status':'complete', 'all_stationarity_pass':all(r['stationarity_pass'] for r in rows),
        'artifacts_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print('Head convergence diagnostic complete; inspect complete.json and stationarity gates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.run,args.output)
