"""Checkpoints bound to recipe, study, projection, scaler and canonical identities."""
import torch
from .config import fingerprint
from .data import digest, write_json
from .models import FrozenRisk
from .training import make_model


def save_checkpoint(path, model, recipe, config, scaler, kind):
    if kind not in {'ssl', 'risk'} or recipe not in config['recipes']:
        raise ValueError('Invalid checkpoint kind/recipe')
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Immutable checkpoint already exists')
    torch.save(model.state_dict(), path)
    write_json(path.with_suffix('.json'), {'format': 'r0_r1_checkpoint_v1', 'recipe': recipe, 'kind': kind,
        'config_sha256': fingerprint(config), 'scaler_sha256': fingerprint(scaler),
        'task_identifier': config['task_identifier'], 'projection': config['projection'],
        'scope': scaler['scope'], 'run_sha256': scaler['run_sha256'],
        'export_sha256': scaler['export_sha256'], 'weights_sha256': digest(path)})


def load_checkpoint(path, recipe, config, scaler, kind, device='cpu'):
    import json
    meta = json.loads(path.with_suffix('.json').read_text())
    expected = {'format': 'r0_r1_checkpoint_v1', 'recipe': recipe, 'kind': kind,
        'config_sha256': fingerprint(config), 'scaler_sha256': fingerprint(scaler),
        'task_identifier': config['task_identifier'], 'projection': config['projection'],
        'scope': scaler['scope'], 'run_sha256': scaler['run_sha256'],
        'export_sha256': scaler['export_sha256'], 'weights_sha256': digest(path)}
    if meta != expected:
        raise ValueError('Checkpoint/config/scaler/source identity mismatch')
    base = make_model(recipe, config, device)
    model = FrozenRisk(base).to(device) if kind == 'risk' else base
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.eval()
