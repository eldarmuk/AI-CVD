import unittest
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.architecture_study.analyze import metrics, paired_bootstrap, representation
from src.architecture_study.saved_analysis import evidence_from_r0


class AnalysisTests(unittest.TestCase):
    def test_saved_r0_evidence_separates_steps_only_from_vitals(self):
        z = np.zeros((3,59), dtype=np.float32)
        z[0,57] = 1
        z[1,37] = .1  # Steps only.
        z[2,33] = .1  # HR observed.
        vital, primitive = evidence_from_r0(z)
        np.testing.assert_array_equal(vital, [False,False,True])
        np.testing.assert_array_equal(primitive, [False,True,True])
        z[1,57] = 1
        with self.assertRaises(ValueError):
            evidence_from_r0(z)

    def test_weighted_ranking_ties_and_threshold(self):
        d = metrics([1,0], [.5,.5], [1,9])
        self.assertAlmostEqual(d['auroc'], .5)
        self.assertAlmostEqual(d['ap'], .1)
        self.assertAlmostEqual(d['precision_at_0_5'], .1)
        self.assertEqual(d['recall_at_0_5'], 1)
        self.assertEqual(metrics([1,0], [.9,.1], [1,9])['auroc'], 1)
        self.assertEqual(metrics([1,0], [.1,.9], [1,9])['auroc'], 0)

    def test_missing_classes_and_no_alerts(self):
        self.assertIsNone(metrics([],[],[])['ap'])
        d = metrics([0,0],[.01,.02],[1,1])
        self.assertIsNone(d['auroc']); self.assertIsNone(d['recall_at_0_5'])
        self.assertIsNone(d['precision_at_0_5'])

    def test_paired_patient_bootstrap_identical(self):
        d = paired_bootstrap([1,0,1,0], [.7,.2,.6,.1], [.7,.2,.6,.1], [1,3,1,3], ['a','a','b','b'], repeats=10)
        self.assertEqual(d['missing_class_resamples'], 0)
        self.assertEqual(d['difference_R1_minus_R0']['ap']['percentile_95_interval'], [0,0])

    def test_representation_collapse(self):
        self.assertEqual(representation(np.zeros((4,3)))['effective_rank'], 0)
        self.assertEqual(representation(np.zeros((4,3)))['constant_dimensions'], 3)

    def test_fictional_complete_control_analysis(self):
        from test_architecture_study import fixture
        from src.architecture_study.cli import run
        from src.architecture_study.analyze import analyze
        from src.architecture_study.config import fingerprint
        from src.architecture_study.data import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, scopes, c, task = fixture(root)
            c = copy.deepcopy(c); c['canonical_run'] = str(root)
            c['sampling'].update(patients_per_batch=2, max_windows_per_patient=16)
            c['resources'].update(batch_size=4, ssl_updates=1, scaler_windows=4, checkpoint_every=1,
                                  checkpoint_batches=1, head_iterations=3)
            output = root/'study'; output.mkdir()
            write_json(output/'study.json', {'config_sha256': fingerprint(c)})
            write_json(output/'canonical_provenance.json', {'run_metadata_sha256': store.run_hash, 'train_export_sha256': store.export_hash})
            run(c, task, store, scopes, output, torch.device('cpu'))
            (output/'analysis-progress.log').write_text('still being written')
            with patch('src.architecture_study.analyze.load_config', return_value=(c, task)):
                analyze(output)
            report = json.loads((output/'analysis/comparison.json').read_text())
            self.assertEqual(len(report['rows']), 3*5*4)
            self.assertEqual(len(report['fold_audits']), 3)
            self.assertTrue((output/'analysis/artifacts_sha256.json').exists())
            hashes = json.loads((output/'analysis/artifacts_sha256.json').read_text())
            self.assertFalse(any(name.endswith('.log') for name in hashes))


if __name__ == '__main__':
    unittest.main()
