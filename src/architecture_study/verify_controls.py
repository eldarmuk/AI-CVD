"""Read-only control audit and pooled diagnostics; no fitting or shard access."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import ROOT, load_config, fingerprint
from .data import digest, aligned_values, write_json
from .saved_analysis import analyze_saved, evidence_from_r0
from .analyze import metrics, paired_bootstrap

MODELS = ('R0', 'R1', 'R0_random_frozen', 'process_only', 'R0_value_access')


def check_prediction(d, refs, labels, weights, vital, primitive):
    aligned_values([r['sample_id'] for r in refs], d['sample_ids'], d['scores'])
    expected = {'patients': [r['patient'] for r in refs], 'labels': labels,
                'weights': weights, 'vital_evidence': vital, 'primitive_evidence': primitive}
    for key, value in expected.items():
        if not np.array_equal(d[key], value):
            raise ValueError(f'Control alignment mismatch: {key}')
    scores = np.asarray(d['scores'])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid probabilities')


def verify(run):
    output = run/'analysis'
    hashes = json.loads((output/'artifacts_sha256.json').read_text())
    for name, expected in hashes.items():
        path = (run/name.replace('\\', '/')).resolve()
        if not path.is_relative_to(run.resolve()) or digest(path) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    covered = {name.replace('\\', '/') for name in hashes}
    required = {str(p.relative_to(run)).replace('\\', '/') for p in output.glob('*')
                if p.is_file() and p.name != 'artifacts_sha256.json'}
    if not required <= covered:
        raise ValueError('Analysis manifest omits artifacts')
    original = analyze_saved(run, write_output=False)
    c, _ = load_config(run/'config.toml')
    provenance = json.loads((output/'analysis_provenance.json').read_text())
    if provenance['study_config_sha256'] != fingerprint(c) or provenance['analysis_source_sha256'] != digest(ROOT/'src/architecture_study/analyze.py'):
        raise ValueError('Analysis implementation/config mismatch')
    comparison = json.loads((output/'comparison.json').read_text())
    rows = []; pooled = {name: [] for name in MODELS}; ys = []; ws = []; ps = []
    reproduced = []; vital_all = []
    for fold in range(3):
        folder = run/f'fold-{fold}'
        refs = json.loads((folder/'assessment_samples.json').read_text())
        fit = json.loads((folder/'head_samples.json').read_text())
        scope = json.loads((folder/'patients.json').read_text())['scope']
        y = [int(r['episode_group'] != ['negative']) for r in refs]
        w = [1/r['inclusion_probability'] for r in refs]
        fw = np.array([1/r['inclusion_probability'] for r in fit])
        with np.load(folder/'R0-embeddings.npz', allow_pickle=False) as z:
            fit_z, held_z = z['fit'], z['held']
            aligned_values([r['sample_id'] for r in fit], z['fit_ids'].tolist(), fit_z)
            aligned_values([r['sample_id'] for r in refs], z['held_ids'].tolist(), held_z)
        vital, primitive = evidence_from_r0(held_z)
        for name in MODELS:
            d = json.loads((output/f'fold-{fold}-{name}-predictions.json').read_text())
            check_prediction(d, refs, y, w, vital, primitive)
            if name in ('R0', 'R1'):
                original_d = json.loads((folder/f'{name}-development.json').read_text())
                if d['scores'] != original_d['scores']:
                    raise ValueError('Original scores changed')
            else:
                head = json.loads((output/f'fold-{fold}-{name}-head.json').read_text())
                h = head['fitting']
                if head['feature_definition'] != name or h['scope'] != scope or not h['encoder_unchanged'] or h['l2'] != c['optimization']['head_l2'] or h['iterations_cap'] != c['resources']['head_iterations'] or h['weighting'] != 'inverse_window_inclusion_probability':
                    raise ValueError('Control head identity mismatch')
                if name != 'R0_random_frozen':
                    columns = slice(32,None) if name == 'process_only' else list(range(32))+[57,58]
                    mean = np.average(fit_z[:,columns].astype(float), axis=0, weights=fw)
                    np.testing.assert_allclose(mean, h['embedding_mean'], rtol=1e-10, atol=1e-10)
                    layer = torch.nn.Linear(held_z[:,columns].shape[1],1)
                    with torch.no_grad():
                        layer.weight.copy_(torch.tensor(head['weight']))
                        layer.bias.copy_(torch.tensor(head['bias']))
                        scores = layer(torch.tensor(held_z[:,columns])).squeeze(-1).sigmoid().numpy()
                    np.testing.assert_array_equal(scores, np.array(d['scores'], dtype=np.float32))
                    reproduced.append(f'{fold}/{name}')
            for stratum, mask in [('all', np.ones(len(y),bool)), ('vital_evidence', vital), ('no_vital_evidence', ~vital), ('no_primitive_evidence', ~primitive)]:
                row = {'fold': fold, 'model': name, 'stratum': stratum,
                       **metrics(np.array(y)[mask], np.array(d['scores'])[mask], np.array(w)[mask])}
                if row not in comparison['rows']:
                    raise ValueError('Recorded metrics mismatch')
                rows.append(row)
            pooled[name].extend(d['scores'])
        ys.extend(y); ws.extend(w); ps.extend([r['patient'] for r in refs]); vital_all.extend(vital.tolist())
    intervals = {}
    for control in MODELS[2:]:
        result = paired_bootstrap(ys, pooled[control], pooled['R0'], ws, ps)
        result['difference_joint_R0_minus_control'] = result.pop('difference_R1_minus_R0')
        intervals[control] = result
    return {'status': 'saved_control_artifacts_verified_with_random_replay_limitation',
            'hashed_artifacts': len(hashes), 'original_audit': original,
            'reproduced_control_heads': reproduced,
            'limitation': 'Random-control embeddings were not saved; its hashes, scope, IDs and metrics are verified, but independent score replay requires a separate private-data pass. Fit IDs/weights are source-bound rather than separately stored in control head files.',
            'rows': rows, 'pooled': {name: metrics(ys, scores, ws) for name,scores in pooled.items()},
            'paired_joint_minus_control': intervals,
            'analysis_manifest_sha256': digest(output/'artifacts_sha256.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(args.run.resolve()) or args.output.exists():
        raise ValueError('Use a new private output inside the study run')
    write_json(args.output, verify(args.run))
    print('Saved control audit complete; no fitting or patient shards accessed.')
