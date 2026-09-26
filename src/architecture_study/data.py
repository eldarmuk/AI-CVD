"""Random indexed TRAIN-only access, label-blind sampling and fold scaling."""
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from src.ai_cvd.dataset import sample_id, patient_split
from src.ai_cvd.features import FEATURE_NAMES
from .config import fingerprint
from .projection import VALUE_INDICES


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def aligned_values(expected_ids, prediction_ids, values):
    """Explicit keyed alignment; duplicate, missing, extra or permuted IDs fail closed."""
    if len(set(expected_ids)) != len(expected_ids) or list(expected_ids) != list(prediction_ids) or len(values) != len(expected_ids):
        raise ValueError('Sample identity/alignment mismatch')
    return values


class TrainingStore:
    def __init__(self, run, task, cache_patients=4, cache_megabytes=256):
        self.run, self.task = Path(run).resolve(), task
        meta_path = self.run / 'run_metadata.json'
        meta = json.loads(meta_path.read_text())
        verification = json.loads((self.run / 'integrity_verification.json').read_text())
        if meta['status'] != 'complete' or meta['task_identifier'] != task.identifier or tuple(meta['feature_names']) != FEATURE_NAMES:
            raise ValueError('Incompatible canonical run')
        self.run_hash = digest(meta_path)
        if verification['status'] != 'passed' or verification['run_metadata_sha256'] != self.run_hash:
            raise ValueError('Missing/stale independent canonical verification')
        export_path = self.run / 'exports/train-stream.json'
        self.export_hash = digest(export_path)
        if meta['artifacts_sha256']['exports/train-stream.json'] != self.export_hash:
            raise ValueError('Training export hash mismatch')
        export = json.loads(export_path.read_text())
        if export['task_identifier'] != task.identifier or tuple(export['feature_names']) != FEATURE_NAMES:
            raise ValueError('Export feature/task mismatch')
        self.shards = export['shards']
        if any(s['split'] != 'train' or patient_split(s['senior_id'], task) != 'train' for s in self.shards):
            raise ValueError('Non-training patient in training export')
        self.by_patient = {s['senior_id']: s for s in self.shards}
        if len(self.by_patient) != len(self.shards):
            raise ValueError('Duplicate training patient')
        self.cache, self.verified = OrderedDict(), {}
        self.cache_patients, self.cache_bytes = cache_patients, cache_megabytes * 1024**2

    def arrays(self, patient, labels=False):
        shard = self.by_patient[patient]
        path = (self.run / shard['file']).resolve()
        if not path.is_relative_to(self.run / 'patients'):
            raise ValueError('Shard path outside canonical patients directory')
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if self.verified.get(patient) != signature:
            if digest(path) != shard['sha256']:
                raise ValueError('Training shard hash mismatch')
            self.verified[patient] = signature
            self.cache.pop(patient, None)
        if patient in self.cache and (not labels or 'target' in self.cache[patient]):
            self.cache.move_to_end(patient)
            return self.cache[patient]
        keys = ['features', 'end_rows', 'sample_ids', 'grid_start_ns']
        if labels:
            keys += ['target', 'first_event', 'event_stop']
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in keys}
        n = len(a['end_rows'])
        if a['features'].shape[1:] != (len(FEATURE_NAMES),) or a['sample_ids'].shape != (n, 32) or n != shard['samples']:
            raise ValueError('Invalid indexed shard shape')
        if np.any(np.diff(a['end_rows'].astype(np.int64)) <= 0) or np.any(a['end_rows'] < self.task.sequence_steps - 1) or np.any(a['end_rows'] >= len(a['features'])):
            raise ValueError('Invalid indexed prefix bounds/order')
        size = sum(v.nbytes for v in a.values())
        self.cache.pop(patient, None)
        while self.cache and (len(self.cache) >= self.cache_patients or sum(v.nbytes for c in self.cache.values() for v in c.values()) + size > self.cache_bytes):
            self.cache.popitem(last=False)
        if size <= self.cache_bytes:
            self.cache[patient] = a
        return a

    def identity(self, patient, index, arrays=None):
        a = self.arrays(patient) if arrays is None else arrays
        if index < 0 or index >= len(a['end_rows']):
            raise ValueError('Sample index out of bounds')
        end = int(a['end_rows'][index])
        prediction = int(a['grid_start_ns']) + (end + 1) * self.task.grid_minutes * 60 * 10**9
        sid = sample_id(patient, datetime.fromtimestamp(prediction / 1e9, timezone.utc))
        if bytes(a['sample_ids'][index]).hex() != sid.removeprefix('sample_'):
            raise ValueError('Immutable sample identity mismatch')
        return sid, prediction

    def batch(self, refs, scope, operation, labels=False):
        patients = [r['patient'] for r in refs]
        scope.authorize(patients, operation)
        if labels and operation not in {'head', 'assessment'}:
            raise ValueError('SSL/scalers/checkpointing must not read window labels')
        X, ids, times, y, episodes = [], [], [], [], []
        for r in refs:
            p, i = r['patient'], int(r['index'])
            a = self.arrays(p, labels)
            sid, prediction = self.identity(p, i, a)
            if r.get('sample_id', sid) != sid:
                raise ValueError('Sampling manifest identity mismatch')
            end = int(a['end_rows'][i])
            X.append(a['features'][end - self.task.sequence_steps + 1:end + 1])
            ids.append(sid); times.append(prediction)
            if labels:
                linked = [e for e in self.by_patient[p]['events'][int(a['first_event'][i]):int(a['event_stop'][i])]]
                if bool(a['target'][i]) != bool(linked) or any(not prediction < e[0] <= prediction + self.task.horizon_minutes * 60 * 10**9 for e in linked):
                    raise ValueError('Target/episode timing mismatch')
                y.append(int(a['target'][i])); episodes.append([e[1] for e in linked])
        aligned_values(ids, ids, X)
        return {'X': np.stack(X), 'sample_ids': ids, 'patients': patients, 'prediction_ns': times,
                'y': np.array(y), 'episodes': episodes, 'scope': scope.identifier, 'operation': operation,
                'task_identifier': self.task.identifier, 'run_sha256': self.run_hash, 'export_sha256': self.export_hash}


class BlockSampler:
    """Uniform patients, then uniform UTC blocks, then uniform windows in a block.

    A fixed label-blind pool caps distinct windows per patient. Cycles sample that
    pool without replacement. Selection probabilities are logged conditionally;
    SSL does not interpret the pool as a natural-prevalence evaluation sample.
    """
    def __init__(self, store, scope, operation, seed, config):
        self.store, self.scope, self.operation = store, scope, operation
        self.rng = np.random.default_rng(seed)
        self.patients = sorted(scope.fit if operation in {'ssl', 'scaler'} else scope.held)
        scope.authorize(self.patients, operation)
        self.config, self.pools, self.cycles = config, {}, {}

    def pool(self, patient):
        if patient not in self.pools:
            a = self.store.arrays(patient)
            times = int(a['grid_start_ns']) + (a['end_rows'].astype(np.int64) + 1) * self.store.task.grid_minutes * 60 * 10**9
            blocks = times // (self.config['block_minutes'] * 60 * 10**9)
            unique, starts, counts = np.unique(blocks, return_index=True, return_counts=True)
            remaining = {int(b): list(range(int(s), int(s + n))) for b, s, n in zip(unique, starts, counts)}
            selected = []
            for _ in range(min(len(times), self.config['max_windows_per_patient'])):
                choices = sorted(remaining)
                b = int(self.rng.choice(choices)); candidates = remaining[b]
                q = 1 / len(choices) / len(candidates)
                offset = int(self.rng.integers(len(candidates))); i = candidates.pop(offset)
                if not candidates:
                    del remaining[b]
                sid, _ = self.store.identity(patient, i, a)
                selected.append({'patient': patient, 'index': i, 'sample_id': sid,
                                 'block': b, 'pool_conditional_probability': q})
            if not selected:
                raise ValueError('Patient has no eligible stream windows')
            self.pools[patient] = selected
        return self.pools[patient]

    def draw(self, batch_size):
        k = self.config['patients_per_batch']
        if batch_size % k or k > len(self.patients):
            raise ValueError('Insufficient patients or incompatible batch size')
        refs = []
        for p in self.rng.choice(self.patients, k, replace=False):
            pool = self.pool(p)
            n = min(batch_size // k, len(pool))
            # Start a new cycle before a visit that would cross its boundary.
            if len(self.cycles.get(p, [])) < n:
                self.cycles[p] = self.rng.permutation(len(pool)).tolist()
            for _ in range(n):
                q = (k / len(self.patients)) / len(self.cycles[p])
                r = dict(pool[self.cycles[p].pop()]); r['draw_conditional_probability'] = q
                refs.append(r)
        return refs


def fit_scaler(store, refs, scope):
    scope.authorize([r['patient'] for r in refs], 'scaler')
    rows = {}
    for r in refs:
        p, i = r['patient'], int(r['index']); a = store.arrays(p)
        sid, _ = store.identity(p, i, a)
        if sid != r['sample_id']:
            raise ValueError('Scaler sample identity mismatch')
        end = int(a['end_rows'][i])
        rows.setdefault(p, set()).update(range(end - store.task.sequence_steps + 1, end + 1))
    count = np.zeros(6, dtype=np.int64); total = np.zeros(6); squared = np.zeros(6)
    population = hashlib.sha256()
    for p in sorted(rows):
        index = sorted(rows[p]); a = store.arrays(p)
        x = a['features'][index][:, VALUE_INDICES].astype(np.float64)
        mask = np.isfinite(x)
        from .projection import PRIMITIVES
        natural = a['features'][index][:, [FEATURE_NAMES.index('observed_'+f) for f in PRIMITIVES]]
        if not np.array_equal(natural, mask.astype(natural.dtype)):
            raise ValueError('Scaler primitive validity mismatch')
        count += mask.sum(0); total += np.where(mask, x, 0).sum(0); squared += np.where(mask, x*x, 0).sum(0)
        population.update(json.dumps([p, index]).encode())
    mean = total / count.clip(1)
    scale = np.sqrt(np.maximum(0, squared / count.clip(1) - mean**2))
    scale[scale < 1e-8] = 1
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'observations': count.tolist(),
            'unobserved_channels': np.flatnonzero(count == 0).tolist(), 'scope': scope.identifier,
            'population': 'label_blind_unique_sampled_training_fold_buckets',
            'row_population_sha256': population.hexdigest(), 'unique_rows': sum(map(len, rows.values())),
            'run_sha256': store.run_hash, 'export_sha256': store.export_hash,
            'projection': 'primitive6_process25_v1', 'task_identifier': store.task.identifier}


def supervised_refs(store, scope, operation, seed, resources):
    """Known-probability strata; positives grouped by linked episode set, not events counted as windows."""
    patients = sorted(scope.fit if operation == 'head' else scope.held)
    scope.authorize(patients, operation)
    rng, refs = np.random.default_rng(seed), []
    for p in patients:
        a = store.arrays(p, labels=True)
        strata = {('negative',): np.flatnonzero(a['target'] == 0).tolist()}
        for i in np.flatnonzero(a['target']):
            key = tuple(e[1] for e in store.by_patient[p]['events'][int(a['first_event'][i]):int(a['event_stop'][i])])
            if not key:
                raise ValueError('Positive without linked episode')
            strata.setdefault(key, []).append(int(i))
        for key, indices in strata.items():
            budget = resources['head_negative_per_patient'] if key == ('negative',) else resources['head_positive_per_episode_group']
            k = min(budget, len(indices))
            for i in rng.choice(indices, k, replace=False):
                sid, _ = store.identity(p, int(i), a)
                refs.append({'patient': p, 'index': int(i), 'sample_id': sid,
                             'inclusion_probability': k / len(indices), 'episode_group': list(key)})
    if len(refs) > resources['head_max_samples']:
        raise ValueError('Supervised sample cap exceeded; explicitly revise study config')
    return refs
