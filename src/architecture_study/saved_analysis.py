"""Short read-only analysis of completed artifacts. No fitting or patient-shard reads."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import ROOT, load_config, fingerprint, grouped_folds
from .data import TrainingStore, digest, write_json
from .resume import verify_training_identity, saved_fold, completed_model
from .analyze import metrics, paired_bootstrap, representation


def evidence_from_r0(z):
    """R0 appends fixed natural-mask means at positions 32:38 (not learned states)."""
    if z.ndim != 2 or z.shape[1] != 59:
        raise ValueError('Unexpected R0 representation contract')
    masks = z[:,32:38]
    if not np.isfinite(z).all() or np.any((masks < 0) | (masks > 1)):
        raise ValueError('Invalid saved natural-mask summaries')
    any_primitive = (masks > 0).any(1)
    if not np.array_equal(~any_primitive, z[:,57].astype(bool)):
        raise ValueError('Saved evidence flag disagrees with natural masks')
    return (masks[:,:5] > 0).any(1), any_primitive


def analyze_saved(run):
    begun = time.perf_counter()
    c, task = load_config(run/'config.toml')
    current, _ = load_config()
    if fingerprint(current) != fingerprint(c):
        raise ValueError('Frozen config changed')
    torch.set_num_threads(4)
    study = verify_training_identity(run, c, 'cpu')
    cli_name = 'src/architecture_study/cli.py'
    original = {k.replace('\\','/'):v for k,v in study['implementation_sha256'].items()}
    if original[cli_name] != digest(ROOT/cli_name):
        receipts = [json.loads(p.read_text()) for p in run.glob('resume-*.json')]
        if not any(r['original_study_sha256'] == digest(run/'study.json') and r['original_cli_sha256'] == original[cli_name]
                   and r['resume_cli_sha256'] == digest(ROOT/cli_name) and r['config_sha256'] == fingerprint(c) for r in receipts):
            raise ValueError('Missing compatible resume provenance')
    if json.loads((run/'complete.json').read_text())['status'] != 'complete':
        raise ValueError('Study incomplete')
    store = TrainingStore(ROOT/c['canonical_run'], task)  # Metadata and TRAIN export only.
    p = json.loads((run/'canonical_provenance.json').read_text())
    if p != {'run_metadata_sha256': store.run_hash, 'train_export_sha256': store.export_hash}:
        raise ValueError('Canonical fingerprints changed')
    scopes = grouped_folds(store.shards, c['seed'], c['folds'])
    rows, audits, learning, pooled_y, pooled_w, pooled_patients = [], [], [], [], [], []
    pooled_scores = {'R0': [], 'R1': []}
    for scope in scopes:
        folder = run/f'fold-{scope.fold}'
        scaler, plan, checkpoints, fit, held = saved_fold(folder, scope, c)
        for operation, groups in [('ssl', plan), ('checkpoint', checkpoints), ('head', [fit]), ('assessment', [held]),
                                  ('scaler', [json.loads((folder/'scaler_samples.json').read_text())])]:
            for refs in groups:
                scope.authorize([r['patient'] for r in refs], operation)
        y = np.array([int(r['episode_group'] != ['negative']) for r in held])
        w = np.array([1/r['inclusion_probability'] for r in held])
        patients = [r['patient'] for r in held]
        if not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError('Invalid sampling weights')
        with np.load(folder/'R0-embeddings.npz', allow_pickle=False) as z:
            vital, primitive = evidence_from_r0(z['held'])
        masks = []
        for recipe in ('R0','R1'):
            mask_hash = completed_model(folder, recipe, c, scaler)
            if mask_hash is None:
                raise ValueError('Missing completed fit')
            masks.append(mask_hash)
            d = json.loads((folder/f'{recipe}-development.json').read_text())
            tel = json.loads((folder/f'{recipe}-telemetry.json').read_text())
            scores = np.asarray(d['scores'])
            for stratum, selected in [('all', np.ones(len(y),bool)), ('vital_evidence', vital),
                                     ('no_vital_evidence', ~vital), ('no_primitive_evidence', ~primitive)]:
                rows.append({'fold': scope.fold, 'model': recipe, 'stratum': stratum, **metrics(y[selected],scores[selected],w[selected])})
            draws = tel['draws']; n = sum(x['windows'] for x in draws)
            with np.load(folder/f'{recipe}-embeddings.npz', allow_pickle=False) as z:
                rep = representation(z['held'])
            learning.append({'fold': scope.fold, 'model': recipe, 'initial_held_loss': tel['initial_held_loss'],
                'best_held_loss': min(x['inner_ssl_loss'] for x in d['history']), 'last_held_loss': d['history'][-1]['inner_ssl_loss'],
                'train_loss_first100_mean': float(np.mean([x['loss'] for x in draws[:100]])),
                'train_loss_last100_mean': float(np.mean([x['loss'] for x in draws[-100:]])),
                'history': d['history'], 'selected_draw': tel['selected_draw'], 'masked_targets': sum(x['targets'] for x in draws),
                'zero_target_window_percent': 100*sum(x['zero_target_windows'] for x in draws)/n,
                'no_primitive_window_percent': 100*sum(x['no_primitive_windows'] for x in draws)/n,
                'window_exposures': n, 'unique_samples': tel['unique_samples'], 'patients': tel['exposed_patients'],
                'optimizer_updates': tel['optimizer_updates'], 'ssl_wall_seconds': tel['ssl_wall_seconds'],
                'windows_per_second': tel['windows_per_second'], 'peak_rss_bytes': tel['process_cumulative_peak_rss_bytes'],
                'head_export_seconds_approx': (folder/f'{recipe}-risk.json').stat().st_mtime-(folder/f'{recipe}-ssl.json').stat().st_mtime,
                'representation': rep, 'head_fitting': d['head']})
            pooled_scores[recipe].extend(scores.tolist())
        if len(set(masks)) != 1:
            raise ValueError('R0/R1 sample and corruption digests differ')
        audits.append({'fold':scope.fold, 'fit_patients':len(scope.fit), 'held_patients':len(scope.held),
            'head_samples':len(fit), 'assessment_samples':len(held), 'positive_windows':int(y.sum()),
            'event_patients':len({p for p,label in zip(patients,y) if label}),
            'linked_episodes':len({e for r in held for e in r['episode_group'] if e != 'negative'}),
            'vital_windows':int(vital.sum()), 'no_vital_windows':int((~vital).sum()), 'no_primitive_windows':int((~primitive).sum()),
            'sampling_plan_sha256':digest(folder/'ssl_sampling.json'), 'sample_corruption_sha256':masks[0]})
        pooled_y.extend(y.tolist()); pooled_w.extend(w.tolist()); pooled_patients.extend(patients)
    result = {'status':'verified_saved_artifacts', 'controls_status':'not fitted/saved; unavailable, no fitting performed',
        'verification_scope':'all six completed model artifacts, source metadata, TRAIN membership and saved plans; no raw database or patient shards reread',
        'folds':audits, 'rows':rows, 'learning':learning,
        'pooled':{k:metrics(pooled_y,v,pooled_w) for k,v in pooled_scores.items()},
        'bootstrap':paired_bootstrap(pooled_y,pooled_scores['R0'],pooled_scores['R1'],pooled_w,pooled_patients),
        'analysis_seconds':time.perf_counter()-begun, 'config_sha256':fingerprint(c), 'task_identifier':task.identifier,
        'run_metadata_sha256':store.run_hash, 'source_sha256':digest(Path(__file__))}
    write_json(run/'saved_artifact_analysis.json', result)
    print('All six fits verified; saved-only paired analysis written. No fitting or patient-shard access.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    analyze_saved(Path(args.run))
