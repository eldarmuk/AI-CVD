"""Predeclared development diagnostics; never loads canonical validation/test."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from .config import ROOT, Scope, load_config, fingerprint
from .data import TrainingStore, aligned_values, digest, write_json
from .artifacts import load_checkpoint
from .training import embed_refs, fit_head, development_metrics
from .projection import VALUE_INDICES


def metrics(y, p, w):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
    if not len(y):
        return {'samples': 0, 'reason': 'empty stratum', 'ap': None, 'auroc': None}
    d = development_metrics(y, p, w)
    pos, neg = np.sum(w*y), np.sum(w*(1-y))
    order = np.argsort(-p, kind='stable'); ys, ps, ws = y[order], p[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps)-1]
    cp, cn = np.cumsum(ws*ys)[ends], np.cumsum(ws*(1-ys))[ends]
    dp, dn = np.diff(np.r_[0, cp]), np.diff(np.r_[0, cn])
    auc = float(np.sum((cp-dp/2)*dn)/(pos*neg)) if pos and neg else None
    alerts = p >= .5
    tp = float(np.sum(w*y*alerts)); emitted = float(np.sum(w*alerts))
    return {'samples': len(y), 'positive_windows': int(y.sum()), 'ap': d['ip_weighted_average_precision'],
            'auroc': auc, 'recall_at_0_5': tp/pos if pos else None,
            'precision_at_0_5': tp/emitted if emitted else None,
            'predicted_positive_samples_at_0_5': int(alerts.sum()),
            'log_loss': d['ip_weighted_log_loss'], 'brier': float(np.sum(w*(p-y)**2)/w.sum()),
            'mean_risk': float(np.sum(w*p)/w.sum()), 'weighted_prevalence': float(pos/w.sum())}


def representation(z):
    z = np.asarray(z, dtype=float)
    centered = z-z.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    q = energy/energy.sum() if energy.sum() else np.zeros_like(energy)
    positive = q[q>0]
    return {'rows': len(z), 'dimensions': z.shape[1], 'finite': bool(np.isfinite(z).all()),
            'per_dimension_sd': z.std(0).tolist(), 'constant_dimensions': int((z.std(0)<1e-8).sum()),
            'effective_rank': float(np.exp(-np.sum(positive*np.log(positive)))) if len(positive) else 0,
            'mean_l2_norm': float(np.linalg.norm(z, axis=1).mean())}


def paired_bootstrap(y, p0, p1, w, patients, seed=20260926, repeats=500):
    patients = np.asarray(patients); unique, index = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed); differences = {'ap': [], 'auroc': []}; missing = 0
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[index]
        valid = multiplicity > 0
        weight = np.asarray(w)[valid]*multiplicity[valid]
        a = metrics(np.asarray(y)[valid], np.asarray(p0)[valid], weight)
        b = metrics(np.asarray(y)[valid], np.asarray(p1)[valid], weight)
        if a['auroc'] is None:
            missing += 1; continue
        for name in differences:
            differences[name].append(b[name]-a[name])
    return {'repeats': repeats, 'missing_class_resamples': missing, 'seed': seed,
            'difference_R1_minus_R0': {k: {'mean': float(np.mean(v)), 'percentile_95_interval': np.quantile(v, [.025, .975]).tolist()} if v else None for k,v in differences.items()},
            'scope': 'paired patient clusters conditional on fixed fitted models and sampled assessment windows'}


def analyze(run):
    c, task = load_config(run/'config.toml')
    study = json.loads((run/'study.json').read_text())
    if fingerprint(c) != study['config_sha256'] or json.loads((run/'complete.json').read_text())['status'] != 'complete':
        raise ValueError('Incomplete or modified study')
    for name, expected in study.get('implementation_sha256', {}).items():
        name = name.replace('\\', '/')
        if digest(ROOT/name) != expected:
            receipts = [json.loads(p.read_text()) for p in run.glob('resume-*.json')]
            allowed_resume = name == 'src/architecture_study/cli.py' and any(
                r['original_study_sha256'] == digest(run/'study.json') and r['original_cli_sha256'] == expected
                and r['resume_cli_sha256'] == digest(ROOT/name) and r['config_sha256'] == fingerprint(c) for r in receipts)
            if not allowed_resume:
                raise ValueError('Training implementation changed without a compatible resume record')
    output = run/'analysis'; output.mkdir(exist_ok=False)
    write_json(output/'analysis_provenance.json', {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               'analysis_source_sha256': digest(Path(__file__)), 'bootstrap_seed': 20260926, 'bootstrap_repeats': 500,
               'diagnostic_probability_threshold': .5, 'study_config_sha256': fingerprint(c)})
    store = TrainingStore(ROOT/c['canonical_run'], task)
    provenance = json.loads((run/'canonical_provenance.json').read_text())
    if store.run_hash != provenance['run_metadata_sha256'] or store.export_hash != provenance['train_export_sha256']:
        raise ValueError('Canonical provenance changed')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    rows, all_y, all_w, all_patients, all_scores = [], [], [], [], {'R0': [], 'R1': []}
    all_vital = []
    fold_audits = []
    for fold in range(3):
        folder = run/f'fold-{fold}'
        saved = json.loads((folder/'patients.json').read_text())
        scope = Scope(frozenset(saved['fit']+saved['held']), frozenset(saved['fit']), frozenset(saved['held']), fold)
        if scope.identifier != saved['scope']:
            raise ValueError('Fold identity mismatch')
        for name, expected in json.loads((folder/'plan_hashes.json').read_text()).items():
            if digest(folder/name) != expected:
                raise ValueError('Sampling plan changed')
        scaler = json.loads((folder/'scaler.json').read_text())
        fit_refs = json.loads((folder/'head_samples.json').read_text())
        held_refs = json.loads((folder/'assessment_samples.json').read_text())
        y, fit_y, vitals, primitive, held_ids = [], [], [], [], []
        for start in range(0, len(held_refs), c['resources']['batch_size']):
            batch = store.batch(held_refs[start:start+c['resources']['batch_size']], scope, 'assessment', labels=True)
            y.extend(batch['y'].tolist()); held_ids.extend(batch['sample_ids'])
            visible = np.isfinite(batch['X'][:,:,VALUE_INDICES])
            vitals.extend(visible[:,:,:5].any((1,2)).tolist()); primitive.extend(visible.any((1,2)).tolist())
        for start in range(0, len(fit_refs), c['resources']['batch_size']):
            batch = store.batch(fit_refs[start:start+c['resources']['batch_size']], scope, 'head', labels=True)
            fit_y.extend(batch['y'].tolist())
        w = np.array([1/r['inclusion_probability'] for r in held_refs])
        fit_w = [1/r['inclusion_probability'] for r in fit_refs]
        y = np.asarray(y); vitals = np.asarray(vitals); primitive = np.asarray(primitive)
        patients = [r['patient'] for r in held_refs]
        predictions = {}; embeddings = {}; shared_mask = None
        for recipe in ('R0', 'R1'):
            d = json.loads((folder/f'{recipe}-development.json').read_text())
            aligned_values(held_ids, d['sample_ids'], d['scores'])
            if not np.array_equal(y, d['labels']):
                raise ValueError('Assessment label mismatch')
            if shared_mask is not None and shared_mask != d['sample_corruption_sha256']:
                raise ValueError('Corruption pairing mismatch')
            shared_mask = d['sample_corruption_sha256']
            predictions[recipe] = np.array(d['scores'])
            with np.load(folder/f'{recipe}-embeddings.npz', allow_pickle=False) as z:
                aligned_values(held_ids, z['held_ids'].tolist(), z['held'])
                aligned_values([r['sample_id'] for r in fit_refs], z['fit_ids'].tolist(), z['fit'])
                embeddings[recipe] = (z['fit'], z['held'])
            exported = load_checkpoint(folder/f'{recipe}-risk.pt', recipe, c, scaler, 'risk')
            pretrained = load_checkpoint(folder/f'{recipe}-ssl.pt', recipe, c, scaler, 'ssl')
            if any(not torch.equal(v, exported.encoder.state_dict()[k]) for k,v in pretrained.encoder.state_dict().items()):
                raise ValueError('Frozen encoder differs from selected SSL checkpoint')
            with torch.no_grad():
                reproduced = exported.head(torch.tensor(embeddings[recipe][1])).squeeze(-1).sigmoid().numpy()
            if not np.array_equal(reproduced, predictions[recipe].astype(np.float32)):
                raise ValueError('Exported head does not reproduce recorded scores')
            write_json(output/f'fold-{fold}-{recipe}-representation.json', {'fit': representation(embeddings[recipe][0]), 'held': representation(embeddings[recipe][1])})
        # Cheap frozen-head controls; no encoder training, new search or label population.
        base = load_checkpoint(folder/'R0-random-ssl.pt', 'R0', c, scaler, 'ssl')
        random_risk, random_fit, _, _, _ = embed_refs(base, store, fit_refs, scope, 'head', scaler, torch.device('cpu'), c['resources']['batch_size'])
        _, random_held, _, _, _ = embed_refs(base, store, held_refs, scope, 'assessment', scaler, torch.device('cpu'), c['resources']['batch_size'])
        controls = {'R0_random_frozen': (random_fit.numpy(), random_held.numpy()),
                    'process_only': (embeddings['R0'][0][:,32:], embeddings['R0'][1][:,32:]),
                    'R0_value_access': (embeddings['R0'][0][:,list(range(32))+[57,58]], embeddings['R0'][1][:,list(range(32))+[57,58]])}
        for name, (fit_z, held_z) in controls.items():
            random_risk.head = torch.nn.Linear(fit_z.shape[1], 1)
            fitted = fit_head(random_risk, torch.tensor(fit_z), torch.tensor(fit_y), fit_w,
                [r['patient'] for r in fit_refs], scope, c['optimization']['head_l2'], c['resources']['head_iterations'])
            with torch.no_grad():
                predictions[name] = random_risk.head(torch.tensor(held_z)).squeeze(-1).sigmoid().numpy()
            write_json(output/f'fold-{fold}-{name}-head.json', {'fitting': fitted, 'weight': random_risk.head.weight.detach().tolist(), 'bias': random_risk.head.bias.detach().tolist(), 'feature_definition': name})
        for name, scores in predictions.items():
            for stratum, select in [('all', np.ones(len(y), bool)), ('vital_evidence', vitals), ('no_vital_evidence', ~vitals), ('no_primitive_evidence', ~primitive)]:
                rows.append({'fold': fold, 'model': name, 'stratum': stratum, **metrics(y[select], scores[select], w[select])})
            write_json(output/f'fold-{fold}-{name}-predictions.json', {'sample_ids': held_ids, 'patients': patients,
                'labels': y.tolist(), 'scores': scores.tolist(), 'weights': w.tolist(), 'vital_evidence': vitals.tolist(), 'primitive_evidence': primitive.tolist()})
        all_y.extend(y.tolist()); all_w.extend(w.tolist()); all_patients.extend(patients); all_vital.extend(vitals.tolist())
        for name in all_scores:
            all_scores[name].extend(predictions[name].tolist())
        fold_audits.append({'fold': fold, 'fit_patients': len(scope.fit), 'held_patients': len(scope.held),
            'head_samples': len(fit_refs), 'assessment_samples': len(held_refs),
            'positive_assessment_windows': int(y.sum()), 'held_positive_patients': len({p for p,label in zip(patients,y) if label}),
            'linked_assessment_episodes': len({e for r in held_refs for e in r['episode_group'] if e != 'negative'}),
            'no_vital_assessment_windows': int((~vitals).sum()), 'no_primitive_assessment_windows': int((~primitive).sum())})
    bootstrap = paired_bootstrap(all_y, all_scores['R0'], all_scores['R1'], all_w, all_patients)
    aggregate = {name: metrics(all_y, scores, all_w) for name,scores in all_scores.items()}
    write_json(output/'comparison.json', {'rows': rows, 'fold_audits': fold_audits, 'pooled': aggregate, 'paired_bootstrap': bootstrap})
    write_json(output/'artifacts_sha256.json', {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob('*')) if p.is_file()})
    print('Training-fold analysis complete. Private reports saved under analysis/.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    analyze(Path(args.run))
