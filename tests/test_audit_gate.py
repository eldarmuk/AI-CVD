"""Public tests use only fictional blocked-audit files."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from src.ai_cvd.audit_gate import verify_blocked_audit
from src.ai_cvd.cli import file_hash
from src.ai_cvd.dataset import patient_split
from src.ai_cvd.features import FEATURE_NAMES
from src.ai_cvd.task import load_task


class BlockedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'fiction.sqlite'
        self.source.write_bytes(b'fictional fingerprint fixture, not a database')
        self.source.with_name(self.source.name+'-wal').write_bytes(b'')
        self.audit = self.root / 'audit'
        self.audit.mkdir()
        self.task = load_task()
        rows = [{'senior_id': 'fiction', 'split': patient_split('fiction', self.task)}]
        stats = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                 for p in (self.source, self.source.with_name(self.source.name+'-wal'))}
        evidence = {'status': 'blocked_source_declarations', 'task_identifier': self.task.identifier,
                    'feature_names': FEATURE_NAMES, 'source_unchanged_during_inspection': True,
                    'source_file_stats_before': stats, 'source_file_stats_after': stats,
                    'source_sha256': {name: file_hash(self.root/name) for name in stats},
                    'counts': {'source_patients': 1, 'preeligibility_split_patients': dict(Counter(r['split'] for r in rows))}}
        files = {'source_evidence.json': json.dumps(evidence),
                 'source-contract.toml': 'clinical_history_policy = "exclude"\n',
                 'coverage.csv': 'senior_id,enrollment_time\n',
                 'coverage_requests.csv': 'senior_id,status\nfiction,pending\n',
                 'source_patient_splits.jsonl': json.dumps(rows[0])+'\n',
                 'dataset_integrity_report.md': 'Blocked; prevalence unavailable.\n',
                 'data_owner_questions.md': 'Fictional owner question fixture.\n'}
        for name, text in files.items():
            (self.audit/name).write_text(text, encoding='utf-8')
        (self.audit/'artifact_checksums.json').write_text(json.dumps({name: file_hash(self.audit/name) for name in files}))

    def test_complete_blocked_audit(self):
        result = verify_blocked_audit(self.audit, self.source, self.task)
        self.assertFalse(result['dataset_generated'])
        self.assertIn('no new full-file hash', result['fingerprint_check'])

    def test_source_change_requires_reverification(self):
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            verify_blocked_audit(self.audit, self.source, self.task)

    def test_report_tampering_is_rejected(self):
        (self.audit/'dataset_integrity_report.md').write_text('Prevalence is zero')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify_blocked_audit(self.audit, self.source, self.task)

    def test_fabricated_coverage_is_rejected(self):
        (self.audit/'coverage.csv').write_text('senior_id,enrollment_time\nfiction,2030-01-01\n')
        with self.assertRaisesRegex(ValueError, 'fabricate'):
            verify_blocked_audit(self.audit, self.source, self.task)
