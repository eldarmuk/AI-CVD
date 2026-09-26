import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.ai_cvd.dataset import patient_split, sample_id
from datetime import datetime, timezone
from src.ai_cvd.features import FEATURE_NAMES
from src.architecture_study.config import load_config, grouped_folds, Scope
from src.architecture_study.data import TrainingStore, BlockSampler, fit_scaler, aligned_values, digest, write_json, supervised_refs
from src.architecture_study.models import MaskedModel, FrozenRisk, masked_huber
from src.architecture_study.projection import project, mask_observed, prefix_window, PRIMITIVES
from src.architecture_study.synthetic import tensors
from src.architecture_study.training import ssl_step, fit_head, development_metrics
from src.architecture_study.artifacts import save_checkpoint, load_checkpoint

torch.set_num_threads(1)


def fixture(root):
    c, task = load_config()
    (root/'patients').mkdir(); (root/'exports').mkdir()
    patients = [f'fictional-{i}' for i in range(100) if patient_split(f'fictional-{i}', task) == 'train'][:9]
    shards = []
    origin = 1761955200000000000
    step = task.grid_minutes*60*10**9
    for p in patients:
        x = tensors(2).numpy().reshape(192, len(FEATURE_NAMES))
        ends = np.arange(95, 192, dtype=np.int32)
        times = origin+(ends.astype(np.int64)+1)*step
        event = origin+170*step
        target = (times < event) & (times+task.horizon_minutes*60*10**9 >= event)
        ids = np.array([list(bytes.fromhex(sample_id(p, datetime.fromtimestamp(int(t)/1e9, timezone.utc)).removeprefix('sample_'))) for t in times], dtype=np.uint8)
        path = root/'patients'/f'{p}.npz'
        np.savez_compressed(path, features=x, end_rows=ends, sample_ids=ids, grid_start_ns=np.array(origin),
                            target=target.astype(np.uint8), first_event=(times>=event).astype(np.int32),
                            event_stop=(times+task.horizon_minutes*60*10**9>=event).astype(np.int32))
        shards.append({'senior_id': p, 'split': 'train', 'file': f'patients/{p}.npz', 'sha256': digest(path),
                       'samples': len(ends), 'events': [[event, f'episode-{p}']], 'flow': {'events_with_eligible_prediction': 1}})
    write_json(root/'exports/train-stream.json', {'shards': shards, 'task_identifier': task.identifier, 'feature_names': FEATURE_NAMES})
    write_json(root/'run_metadata.json', {'status': 'complete', 'task_identifier': task.identifier,
        'feature_names': FEATURE_NAMES, 'artifacts_sha256': {'exports/train-stream.json': digest(root/'exports/train-stream.json')}})
    write_json(root/'integrity_verification.json', {'status': 'passed', 'run_metadata_sha256': digest(root/'run_metadata.json')})
    return TrainingStore(root, task), grouped_folds(shards, c['seed']), c, task


class ModelGates(unittest.TestCase):
    def setUp(self):
        self.c, self.task = load_config()
        torch.manual_seed(1)

    def view(self, x, A=None):
        return project(x, np.zeros(6), np.ones(6), self.task, A)

    def test_config_and_unimplemented_recipes(self):
        self.assertEqual(self.task.grid_minutes*self.task.sequence_steps, 480)
        for recipe in ('R2', 'R3', 'R4'):
            with self.assertRaises(ValueError):
                MaskedModel(recipe)

    def test_empty_safe_null_pool_head_and_zero_loss(self):
        v = self.view(tensors(2, observed_fraction=0))
        for recipe in ('R0', 'R1'):
            m = MaskedModel(recipe)
            self.assertTrue(torch.equal(m.encoder.value(v), torch.zeros(2, 32)))
            z = m.encoder(v)
            self.assertTrue(torch.equal(z[:, -2], torch.ones(2)))
            self.assertTrue(torch.isfinite(FrozenRisk(m)(v)).all())
            loss = masked_huber(m(v), v)
            self.assertEqual(float(loss), 0)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            before = copy.deepcopy(m.state_dict())
            loss, targets = ssl_step(m, torch.optim.Adam(m.parameters()), v)
            self.assertEqual(targets, 0)
            self.assertTrue(all(torch.equal(before[k], val) for k, val in m.state_dict().items()))

    def test_single_observed_zero_steps_and_unknown_recency(self):
        x = tensors(1, observed_fraction=0)
        x[0, 5, FEATURE_NAMES.index('steps')] = 0
        x[0, 5, FEATURE_NAMES.index('observed_steps')] = 1
        v = self.view(x)
        self.assertFalse(mask_observed(v.M, torch.Generator().manual_seed(1)).any())
        self.assertTrue(v.M[0, 5, 5]); self.assertEqual(float(v.values[0, 5, 5]), 0)
        y = x.clone(); y[0, 5, FEATURE_NAMES.index('time_since_last_steps')] = 0
        w = self.view(y)
        self.assertEqual(float(v.process[0, 5, 11]), 0)
        self.assertEqual(float(w.process[0, 5, 11]), 1)
        self.assertEqual(float(w.process[0, 5, 17]), 0)

    def test_hidden_value_cannot_enter_cache_or_reset_age(self):
        x = tensors(1, observed_fraction=0)
        for j in (2, 5):
            x[0, j, FEATURE_NAMES.index('heartrate')] = j
            x[0, j, FEATURE_NAMES.index('observed_heartrate')] = 1
        A = torch.zeros(1, 96, 6, dtype=torch.bool); A[0, 5, 1] = True
        v = self.view(x, A)
        y = x.clone(); y[0, 5, FEATURE_NAMES.index('heartrate')] = 999999
        for recipe in ('R0', 'R1'):
            model = MaskedModel(recipe)
            self.assertTrue(torch.equal(model.encoder(v), model.encoder(self.view(y, A))))
            _, trace = model.encoder.value(v, trace=True)
            self.assertEqual(float(trace['cache'][0, 5, 1]), 2)
            self.assertEqual(float(trace['age'][0, 5, 1]), 15)

    def test_derived_shortcuts_and_feature_order(self):
        x = tensors(2); y = x.clone()
        allowed = set(PRIMITIVES) | {'observed_'+f for f in PRIMITIVES} | {'time_since_last_'+f for f in PRIMITIVES} | {'hour_sin', 'hour_cos', 'is_night', 'steps_delta_interval_minutes', 'observed_steps_source', 'steps_counter_reset'}
        for i, name in enumerate(FEATURE_NAMES):
            if name not in allowed:
                y[..., i] = 123456
        for recipe in ('R0', 'R1'):
            model = MaskedModel(recipe)
            self.assertTrue(torch.equal(model.encoder(self.view(x)), model.encoder(self.view(y))))
        with self.assertRaises(ValueError):
            project(x, np.zeros(6), np.ones(6), self.task, feature_names=FEATURE_NAMES[::-1])

    def test_mask_loss_balance_and_missing_gradients(self):
        x = tensors(1, observed_fraction=0)
        for j, channel in ((0, 0), (1, 0), (2, 1)):
            x[0, j, FEATURE_NAMES.index(PRIMITIVES[channel])] = 0
            x[0, j, FEATURE_NAMES.index('observed_'+PRIMITIVES[channel])] = 1
        x.requires_grad_()
        A = self.view(x).M.clone(); v = self.view(x, A)
        pred = torch.zeros(1, 96, 6, requires_grad=True)
        with torch.no_grad():
            pred[0, 0:2, 0] = 1; pred[0, 2, 1] = 2
        loss = masked_huber(pred, v)
        self.assertAlmostEqual(float(loss), 1.0)  # channel means: .5 and 1.5
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(pred.grad[~A].abs().sum()), 0)
        self.assertEqual(float(x.grad[0, 3].abs().sum()), 0)

    def test_masking_partition_and_encoder_missing_gradients(self):
        x = tensors(2).requires_grad_(); v = self.view(x)
        A = mask_observed(v.M, torch.Generator().manual_seed(42))
        v = self.view(x, A)
        self.assertFalse((A & ~v.M).any()); self.assertTrue(torch.equal(v.V, v.M & ~A))
        m = MaskedModel('R1'); m.encoder(v).sum().backward()
        for i, name in enumerate(PRIMITIVES):
            grad = x.grad[..., FEATURE_NAMES.index(name)]
            self.assertEqual(float(grad[~v.V[..., i]].abs().sum()), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_future_suffix_invariance_and_prefix_validation(self):
        x = tensors(2).numpy().reshape(192, len(FEATURE_NAMES))
        step = 5*60*10**9; times = np.arange(192)*step; cutoff = 96*step
        prefix = prefix_window(x, times, cutoff, self.task)
        y = x.copy(); y[96:] = 99999
        self.assertTrue(np.array_equal(prefix, prefix_window(y, times, cutoff, self.task), equal_nan=True))
        for recipe in ('R0', 'R1'):
            risk = FrozenRisk(MaskedModel(recipe))
            self.assertTrue(torch.equal(risk(self.view(torch.tensor(prefix[None]))), risk(self.view(torch.tensor(prefix_window(y, times, cutoff, self.task)[None])))))
        with self.assertRaises(ValueError):
            prefix_window(x[1:], times[1:], cutoff, self.task)

    def test_tiny_optimization_and_frozen_head(self):
        scope = Scope(frozenset({'a', 'b', 'c'}), frozenset({'a', 'b'}), frozenset({'c'}), 0)
        for recipe in ('R0', 'R1'):
            model = MaskedModel(recipe); x = tensors(4)
            v = self.view(x); v = self.view(x, mask_observed(v.M, torch.Generator().manual_seed(2)))
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            losses = [ssl_step(model, optimizer, v)[0] for _ in range(4)]
            self.assertLess(losses[-1], losses[0])
            risk = FrozenRisk(model); before = copy.deepcopy(risk.encoder.state_dict())
            self.assertFalse(hasattr(risk, 'decoder'))
            z = risk.embeddings(self.view(x))
            result = fit_head(risk, z, torch.tensor([0, 0, 1, 1]), [1]*4, ['a', 'a', 'b', 'b'], scope, iterations=15)
            self.assertLess(result['final_objective'], result['initial_objective'])
            self.assertTrue(all(torch.equal(value, risk.encoder.state_dict()[key]) for key, value in before.items()))
            with self.assertRaises(ValueError):
                risk(v)

    def test_save_load_model_config_identity_and_inference(self):
        scaler = {'scope': 'fictional-fold', 'run_sha256': 'fictional-run', 'export_sha256': 'fictional-export'}
        v = self.view(tensors(2))
        with tempfile.TemporaryDirectory() as temporary:
            for recipe in ('R0', 'R1'):
                model = MaskedModel(recipe)
                for kind, candidate in (('ssl', model), ('risk', FrozenRisk(model))):
                    path = Path(temporary)/f'{recipe}-{kind}.pt'
                    save_checkpoint(path, candidate, recipe, self.c, scaler, kind)
                    loaded = load_checkpoint(path, recipe, self.c, scaler, kind)
                    self.assertTrue(torch.equal(candidate(v), loaded(v)))
                    self.assertTrue(torch.equal(loaded(v), loaded(v)))
                    wrong = copy.deepcopy(self.c); wrong['seed'] += 1
                    with self.assertRaises(ValueError):
                        load_checkpoint(path, recipe, wrong, scaler, kind)
                    with self.assertRaises(ValueError):
                        load_checkpoint(path, recipe, self.c, dict(scaler, scope='other'), kind)
                    with self.assertRaises(FileExistsError):
                        save_checkpoint(path, candidate, recipe, self.c, scaler, kind)


class DataGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.store, self.scopes, self.c, self.task = fixture(self.root)
        self.scope = self.scopes[0]
        self.config = dict(self.c['sampling'], patients_per_batch=2, max_windows_per_patient=32)

    def tearDown(self):
        self.tmp.cleanup()

    def refs(self):
        return BlockSampler(self.store, self.scope, 'ssl', 12, self.config).draw(4)

    def test_patient_folds_and_all_fit_operations(self):
        for scope in self.scopes:
            self.assertFalse(scope.fit & scope.held)
            for operation in ('ssl', 'scaler', 'head', 'reference_density'):
                with self.assertRaises(ValueError):
                    scope.authorize(scope.held, operation)
                for split in ('validation', 'test'):
                    with self.assertRaises(ValueError):
                        scope.authorize(scope.fit, operation, split)
            with self.assertRaises(ValueError):
                scope.authorize(scope.fit, 'checkpoint')
        self.assertEqual(self.scopes, grouped_folds(self.store.shards, self.c['seed']))

    def test_sampler_stability_caps_and_label_blindness(self):
        one = BlockSampler(self.store, self.scope, 'ssl', 12, self.config)
        two = BlockSampler(self.store, self.scope, 'ssl', 12, self.config)
        self.assertEqual(one.draw(4), two.draw(4))
        for _ in range(5):
            refs = one.draw(4)
            self.assertEqual(len({r['sample_id'] for r in refs}), 4)
        self.assertTrue(all(len(pool) <= 32 for pool in one.pools.values()))
        self.assertTrue(all('target' not in a for a in self.store.cache.values()))

    def test_short_patient_pool_preserved_without_duplicates(self):
        sampler = BlockSampler(self.store, self.scope, 'ssl', 2, dict(self.config, max_windows_per_patient=1))
        refs = sampler.draw(8)
        self.assertEqual(len(refs), 2)
        self.assertEqual(len({r['sample_id'] for r in refs}), 2)

    def test_identity_loader_scaler_and_permutation(self):
        refs = self.refs(); batch = self.store.batch(refs, self.scope, 'ssl')
        self.assertEqual(batch['X'].shape, (4, 96, len(FEATURE_NAMES)))
        scaler = fit_scaler(self.store, refs, self.scope)
        duplicate = fit_scaler(self.store, refs+refs, self.scope)
        self.assertEqual(scaler, duplicate)
        rows = {}
        for r in refs:
            end = int(self.store.arrays(r['patient'])['end_rows'][r['index']])
            rows.setdefault(r['patient'], set()).update(range(end-95, end+1))
        values = np.concatenate([self.store.arrays(p)['features'][sorted(index)][:, [FEATURE_NAMES.index(f) for f in PRIMITIVES]] for p, index in rows.items()]).astype(float)
        np.testing.assert_allclose(scaler['mean'], np.nanmean(values, axis=0))
        np.testing.assert_allclose(scaler['scale'], np.nanstd(values, axis=0))
        with self.assertRaises(ValueError):
            aligned_values(batch['sample_ids'], batch['sample_ids'][::-1], [1]*4)
        with self.assertRaises(ValueError):
            self.store.batch([dict(refs[0], sample_id='wrong')], self.scope, 'ssl')
        with self.assertRaises(ValueError):
            self.store.batch(refs, self.scope, 'ssl', labels=True)
        with self.assertRaises(ValueError):
            fit_scaler(self.store, refs, self.scopes[1])

    def test_supervised_episode_linkage_and_probabilities(self):
        refs = supervised_refs(self.store, self.scope, 'head', 1, self.c['resources'])
        batch = self.store.batch(refs, self.scope, 'head', labels=True)
        self.assertEqual(set(batch['y']), {0, 1})
        self.assertTrue(all(0 < r['inclusion_probability'] <= 1 for r in refs))
        self.assertTrue(all(bool(y) == bool(es) for y, es in zip(batch['y'], batch['episodes'])))

    def test_tampered_shard_fails(self):
        p = self.refs()[0]['patient']; path = self.root/self.store.by_patient[p]['file']
        with open(path, 'ab') as f:
            f.write(b'bad')
        with self.assertRaises(ValueError):
            self.store.arrays(p)

    def test_weighted_ap_ties(self):
        m = development_metrics([1, 0], [.5, .5], [1, 9])
        self.assertAlmostEqual(m['ip_weighted_average_precision'], .1)

    def test_tiny_synthetic_runner_and_paired_corruptions(self):
        from src.architecture_study.cli import run
        c = copy.deepcopy(self.c)
        c['sampling'] = self.config
        c['resources'].update(batch_size=4, ssl_updates=2, scaler_windows=8,
                              checkpoint_every=1, checkpoint_batches=1, head_iterations=3)
        output = self.root/'study'; output.mkdir()
        run(c, self.task, self.store, self.scopes, output, torch.device('cpu'))
        self.assertEqual(json.loads((output/'complete.json').read_text())['status'], 'complete')
        for fold in range(3):
            a = json.loads((output/f'fold-{fold}/R0-development.json').read_text())
            b = json.loads((output/f'fold-{fold}/R1-development.json').read_text())
            self.assertEqual(a['sample_corruption_sha256'], b['sample_corruption_sha256'])
            self.assertEqual(a['sample_ids'], b['sample_ids'])
            self.assertTrue(a['head']['encoder_unchanged'] and b['head']['encoder_unchanged'])


if __name__ == '__main__':
    unittest.main()
