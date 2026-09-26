"""Fail-closed study configuration and canonical-training-only access scopes."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tomllib
from src.ai_cvd.task import load_task

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/model_studies/r0_r1_v1.toml'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path=DEFAULT_CONFIG):
    with open(path, 'rb') as f:
        c = tomllib.load(f)
    task = load_task(ROOT / c['task_config'])
    if task.identifier != c['task_identifier']:
        raise ValueError('Canonical task fingerprint mismatch')
    expected = {'version': '1.0.0', 'projection': 'primitive6_process25_v1',
                'pretraining_population': 'train_stream', 'folds': 3,
                'fold_policy': 'training_patients_stratified_hash_round_robin',
                'normalization': 'label_blind_unique_sampled_training_fold_buckets'}
    if any(c.get(k) != v for k, v in expected.items()) or c['recipes'] != ['R0', 'R1']:
        raise ValueError('Unsupported study definition')
    if c['masking'] != {'policy': 'observed_cells_only', 'fraction': 0.2,
                        'minimum_visible': 1, 'minimum_targets': 1}:
        raise ValueError('Unsupported masking definition')
    if c['model']['pooling'] != 'evidence_masked_attention':
        raise ValueError('Unsupported pooling')
    if c['sampling']['policy'] != 'patient_first_utc_blocks_without_replacement_per_cycle':
        raise ValueError('Unsupported sampling policy')
    if c['sampling']['short_pool_policy'] != 'use_all_without_duplication':
        raise ValueError('Unsupported short-patient policy')
    for section in ('model', 'resources', 'sampling', 'optimization'):
        for key, value in c[section].items():
            if isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f'Nonpositive {section}.{key}')
    if c['resources']['batch_size'] % c['sampling']['patients_per_batch']:
        raise ValueError('Batch size must divide into patient visits')
    if c['sampling']['block_minutes'] % task.grid_minutes:
        raise ValueError('Block width must align to canonical grid')
    return c, task


@dataclass(frozen=True)
class Scope:
    """Inner validation is a subset of canonical TRAIN, never canonical validation."""
    training: frozenset
    fit: frozenset
    held: frozenset
    fold: int

    def __post_init__(self):
        if not self.fit or not self.held or self.fit & self.held or self.fit | self.held != self.training:
            raise ValueError('Invalid patient-isolated training fold')

    @property
    def identifier(self):
        return fingerprint({'fold': self.fold, 'fit': sorted(self.fit), 'held': sorted(self.held)})

    def authorize(self, patients, operation, split='train'):
        fitting = {'ssl', 'scaler', 'head', 'reference_density'}
        if split != 'train' or operation not in fitting | {'checkpoint', 'assessment'}:
            raise ValueError('Only scoped canonical-training patients are permitted')
        allowed = self.fit if operation in fitting else self.held
        if not set(patients).issubset(allowed):
            raise ValueError(f'Patient leakage into {operation}')


def grouped_folds(shards, seed, count=3):
    """Only training-patient event-presence flags stratify deterministic assignment."""
    if any(s['split'] != 'train' for s in shards):
        raise ValueError('Non-training patient in fold construction')
    ids = [s['senior_id'] for s in shards]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate patient')
    assignment = {}
    for positive in (False, True):
        group = [s['senior_id'] for s in shards
                 if bool(s['flow'].get('events_with_eligible_prediction', 0)) == positive]
        group.sort(key=lambda sid: fingerprint([seed, sid]))
        if positive and len(group) < count:
            raise ValueError('Too few positive training patients for grouped folds')
        for i, sid in enumerate(group):
            assignment[sid] = i % count
    population = frozenset(ids)
    return [Scope(population, frozenset(s for s in ids if assignment[s] != fold),
                  frozenset(s for s in ids if assignment[s] == fold), fold)
            for fold in range(count)]


def private_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / 'runs/model_studies').resolve()):
        raise ValueError('Study outputs must remain under ignored runs/model_studies/')
    path.mkdir(parents=True, exist_ok=False)
    return path
