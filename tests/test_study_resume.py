import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from test_architecture_study import fixture
from src.architecture_study.cli import run
from src.architecture_study.data import digest
from src.architecture_study.resume import completed_model, verify_training_identity, record_resume
from src.architecture_study.config import load_config, fingerprint, ROOT


class ResumeTests(unittest.TestCase):
    def test_windows_provenance_paths_and_frozen_identity(self):
        c, _ = load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {'config_sha256': fingerprint(c), 'environment': {'device': 'cpu', 'threads': 4},
                'implementation_sha256': {'src\\architecture_study\\cli.py': 'original-orchestration',
                    'src\\architecture_study\\models.py': digest(ROOT/'src/architecture_study/models.py')}}
            (root/'study.json').write_text(json.dumps(metadata))
            verify_training_identity(root, c, 'cpu')
            record_resume(root, c, 'cpu')
            receipt = json.loads(next(root.glob('resume-*.json')).read_text())
            self.assertEqual(receipt['original_cli_sha256'], 'original-orchestration')
            with self.assertRaisesRegex(ValueError, 'config/device/thread'):
                verify_training_identity(root, c, 'cuda')
            metadata['implementation_sha256']['src\\architecture_study\\models.py'] = 'changed'
            (root/'study.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'Frozen implementation changed'):
                verify_training_identity(root, c, 'cpu')

    def test_resume_skips_valid_work_and_preserves_all_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, scopes, c, task = fixture(root)
            c = copy.deepcopy(c)
            c['sampling'].update(patients_per_batch=2, max_windows_per_patient=16)
            c['resources'].update(batch_size=4, ssl_updates=1, scaler_windows=4, checkpoint_every=1, checkpoint_batches=1, head_iterations=3)
            output = root/'study'; output.mkdir()
            # Simulate user pause after a completely saved R0, before R1 starts.
            from src.architecture_study.training import make_model as original
            def stop_before_r1(recipe, config, device):
                if recipe == 'R1':
                    raise RuntimeError('synthetic pause')
                return original(recipe, config, device)
            with patch('src.architecture_study.cli.make_model', side_effect=stop_before_r1):
                with self.assertRaisesRegex(RuntimeError, 'synthetic pause'):
                    run(c, task, store, scopes, output, torch.device('cpu'))
            self.assertTrue((output/'fold-0/R0-complete.json').exists())
            protected = {p: digest(p) for p in output.rglob('*') if p.is_file()}
            run(c, task, store, scopes, output, torch.device('cpu'), resume=True)
            self.assertTrue(all(digest(p) == h for p,h in protected.items()))
            all_outputs = {p:digest(p) for p in output.rglob('*') if p.is_file()}
            with patch('src.architecture_study.cli.ssl_step', side_effect=AssertionError('Must not retrain')):
                run(c, task, store, scopes, output, torch.device('cpu'), resume=True)
            self.assertTrue(all(digest(p) == h for p,h in all_outputs.items()))
            # The resumed path must match an uninterrupted run, not merely finish.
            baseline = root/'uninterrupted'; baseline.mkdir()
            run(c, task, store, scopes, baseline, torch.device('cpu'))
            for fold in range(3):
                for recipe in ('R0', 'R1'):
                    a = json.loads((output/f'fold-{fold}/{recipe}-development.json').read_text())
                    b = json.loads((baseline/f'fold-{fold}/{recipe}-development.json').read_text())
                    self.assertEqual(a['scores'], b['scores'])
                    self.assertEqual(a['sample_corruption_sha256'], b['sample_corruption_sha256'])
            folder = output/'fold-0'; scaler = json.loads((folder/'scaler.json').read_text())
            # Explicit adoption supports the original runner's completed artifact layout.
            (folder/'R0-complete.json').unlink()
            with self.assertRaisesRegex(ValueError, 'explicit validated adoption'):
                completed_model(folder, 'R0', c, scaler)
            self.assertIsNotNone(completed_model(folder, 'R0', c, scaler, adopt=True))
            with open(folder/'R0-risk.pt', 'ab') as f:
                f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                completed_model(folder, 'R0', c, scaler)

    def test_partial_fit_is_never_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'R1-ssl.pt').write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError, 'Partial fit'):
                completed_model(root, 'R1', {}, {})


if __name__ == '__main__':
    unittest.main()
