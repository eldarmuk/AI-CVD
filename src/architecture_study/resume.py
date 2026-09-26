"""Resume only fully verified fits; never overwrite or silently repeat partial work."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from .config import ROOT, fingerprint
from .data import digest, aligned_values, write_json
from .artifacts import load_checkpoint


def implementation_hashes(study):
    """Windows-produced manifests use backslashes; normalize only path separators."""
    return {name.replace('\\', '/'): value for name, value in study['implementation_sha256'].items()}


def verify_training_identity(run, config, device):
    study = json.loads((run/'study.json').read_text())
    if study['config_sha256'] != fingerprint(config) or study['environment']['device'] != str(device) or study['environment']['threads'] != config['resources']['cpu_threads']:
        raise ValueError('Resume config/device/thread mismatch')
    # Only the orchestration file changed to support resume. All original numerical,
    # data, projection, sampling, fitting and checkpoint implementations must match.
    for name, expected in implementation_hashes(study).items():
        if name != 'src/architecture_study/cli.py' and digest(ROOT/name) != expected:
            raise ValueError(f'Frozen implementation changed: {name}')
    return study


def record_resume(run, config, device):
    study = verify_training_identity(run, config, device)
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(run/f'resume-{now}.json', {'format': 'verified_completed_fits_resume_v1',
        'original_study_sha256': digest(run/'study.json'), 'config_sha256': fingerprint(config),
        'original_cli_sha256': implementation_hashes(study)['src/architecture_study/cli.py'],
        'resume_cli_sha256': digest(ROOT/'src/architecture_study/cli.py'),
        'resume_adapter_sha256': digest(Path(__file__)),
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'policy': 'skip verified completed fits; refuse partial fits; retain saved plans and original study metadata'})


def completed_model(folder, recipe, config, scaler, adopt=False):
    names = [f'{recipe}-{suffix}' for suffix in ('ssl.pt','ssl.json','risk.pt','risk.json','development.json','telemetry.json','embeddings.npz')]
    if recipe == 'R0':
        names += ['R0-random-ssl.pt', 'R0-random-ssl.json']
    present = [(folder/name).exists() for name in names]
    if not any(present):
        if (folder/f'{recipe}-random-ssl.pt').exists():
            raise ValueError('Partial fit exists; automatic restart is forbidden')
        return None
    if not all(present):
        raise ValueError('Partial fit exists; automatic restart is forbidden')
    marker = folder/f'{recipe}-complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['status'] != 'complete_verified' or saved['config_sha256'] != fingerprint(config) or saved['scope'] != scaler['scope']:
            raise ValueError('Completion marker identity mismatch')
        if saved['recipe'] != recipe or set(saved['artifacts_sha256']) != set(names + ['plan_hashes.json','scaler.json']):
            raise ValueError('Incomplete completion-marker manifest')
        for name, expected in saved['artifacts_sha256'].items():
            if digest(folder/name) != expected:
                raise ValueError('Completed artifact changed')
    elif not adopt:
        raise ValueError('Completion marker missing; explicit validated adoption required')
    for name, expected in json.loads((folder/'plan_hashes.json').read_text()).items():
        if digest(folder/name) != expected:
            raise ValueError('Saved sample plan changed')
    ssl = load_checkpoint(folder/f'{recipe}-ssl.pt', recipe, config, scaler, 'ssl')
    risk = load_checkpoint(folder/f'{recipe}-risk.pt', recipe, config, scaler, 'risk')
    if any(not torch.equal(v, risk.encoder.state_dict()[k]) for k,v in ssl.encoder.state_dict().items()):
        raise ValueError('Frozen encoder changed')
    d = json.loads((folder/f'{recipe}-development.json').read_text())
    telemetry = json.loads((folder/f'{recipe}-telemetry.json').read_text())
    fit = json.loads((folder/'head_samples.json').read_text())
    held = json.loads((folder/'assessment_samples.json').read_text())
    held_ids = [r['sample_id'] for r in held]
    aligned_values(held_ids, d['sample_ids'], d['scores'])
    if [int(r['episode_group'] != ['negative']) for r in held] != d['labels']:
        raise ValueError('Supervised manifest/label mismatch')
    with np.load(folder/f'{recipe}-embeddings.npz', allow_pickle=False) as z:
        aligned_values(held_ids, z['held_ids'].tolist(), z['held'])
        aligned_values([r['sample_id'] for r in fit], z['fit_ids'].tolist(), z['fit'])
        if not np.isfinite(z['fit']).all() or not np.isfinite(z['held']).all():
            raise ValueError('Nonfinite embeddings')
        with torch.no_grad():
            reproduced = risk.head(torch.tensor(z['held'])).squeeze(-1).sigmoid().numpy()
        if not np.array_equal(reproduced, np.asarray(d['scores'], dtype=np.float32)):
            raise ValueError('Saved scores do not match exported head')
    budget = config['resources']['ssl_updates']
    if [x['draw'] for x in telemetry['draws']] != list(range(1, budget+1)):
        raise ValueError('Incomplete SSL exposure history')
    if telemetry['optimizer_updates'] != sum(x['targets'] > 0 for x in telemetry['draws']):
        raise ValueError('SSL update accounting mismatch')
    if not np.isfinite([x['loss'] for x in telemetry['draws']]).all():
        raise ValueError('Nonfinite SSL history')
    if d['history'][-1]['draw'] != budget or not d['head']['encoder_unchanged']:
        raise ValueError('Fit not complete')
    if marker.exists() and saved['sample_corruption_sha256'] != d['sample_corruption_sha256']:
        raise ValueError('Completion marker corruption identity mismatch')
    if not marker.exists():
        write_json(marker, {'status': 'complete_verified', 'recipe': recipe, 'scope': scaler['scope'],
            'config_sha256': fingerprint(config), 'sample_corruption_sha256': d['sample_corruption_sha256'],
            'artifacts_sha256': {name:digest(folder/name) for name in names + ['plan_hashes.json','scaler.json']},
            'verified_utc': datetime.now(timezone.utc).isoformat()})
    return d['sample_corruption_sha256']


def saved_fold(folder, scope, config):
    saved = json.loads((folder/'patients.json').read_text())
    if saved != {'fit': sorted(scope.fit), 'held': sorted(scope.held), 'scope': scope.identifier}:
        raise ValueError('Saved fold assignment changed')
    hashes = json.loads((folder/'plan_hashes.json').read_text())
    for name, expected in hashes.items():
        if digest(folder/name) != expected:
            raise ValueError('Saved sampling plan changed')
    read = lambda name: json.loads((folder/name).read_text())
    scaler = read('scaler.json')
    if scaler['scope'] != scope.identifier:
        raise ValueError('Saved scaler scope mismatch')
    plan = read('ssl_sampling.json')
    if len(plan) != config['resources']['ssl_updates']:
        raise ValueError('Saved exposure budget mismatch')
    return scaler, plan, read('checkpoint_samples.json'), read('head_samples.json'), read('assessment_samples.json')
