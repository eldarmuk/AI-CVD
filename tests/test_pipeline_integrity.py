"""Synthetic adapter, artifact identity, and adversarial timing regression checks."""
from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.ai_cvd.task import load_task, Task
from src.ai_cvd.features import FEATURE_NAMES, build_features, validate_source_contract
from src.ai_cvd.episodes import iso, as_time, build_episodes
from src.ai_cvd.dataset import generate_patient_manifest, sequence_for_sample, training_manifest
from src.ai_cvd.cli import SQLiteSource, DuckDBSource, build_run, file_hash, localize, verify_run
from src.ai_cvd.synthetic import fixture
from src.ai_cvd.arrays import export_sequences, verified_shards


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        cls.source, cls.coverages, cls.histories, cls.contract = fixture(cls.task)
        cls.c = cls.coverages[0]
        cls.sid = cls.c['senior_id']
        cls.start = as_time(cls.c['enrollment_time'])

    def features(self, readings, contract=None):
        return list(build_features(readings, self.c, [], self.task, contract or self.contract))

    def measurement(self, minute, kind='Steps', **values):
        return dict(senior_id=self.sid, date=iso(self.start+timedelta(minutes=minute)), type=kind, **values)

    def test_pulse_pressure_requires_a_valid_raw_pair(self):
        rows = self.features([self.measurement(3, 'BloodPressure', sbp=120, dbp=None),
                              self.measurement(3, 'BloodPressure', sbp=None, dbp=80)])
        self.assertEqual((rows[0]['sbp'], rows[0]['dbp']), (120, 80))
        self.assertIsNone(rows[0]['pulse_pressure'])
        self.assertEqual(rows[0]['observed_pulse_pressure'], 0)

    def test_daily_counter_boundary_even_without_visible_decrease(self):
        contract = dict(self.contract, steps_semantics='cumulative_counter', counter_reset_policy='daily')
        rows = self.features([self.measurement(1438, value=100), self.measurement(1443, value=110),
                              self.measurement(1448, value=110), self.measurement(1453, value=-1),
                              self.measurement(1458, value=115)], contract)
        self.assertEqual([r['steps'] for r in rows[287:292]], [None, None, 0, None, 5])
        self.assertEqual(rows[288]['steps_counter_reset'], 1)
        self.assertEqual(rows[291]['steps_delta_interval_minutes'], 10)
        self.assertEqual(rows[290]['time_since_last_steps'], 7)

    def test_counter_requires_reset_declaration(self):
        with self.assertRaises(ValueError):
            validate_source_contract(dict(self.contract, steps_semantics='cumulative_counter'))

    def test_source_declarations_cannot_be_placeholders(self):
        for field in ('steps_evidence', 'availability_evidence', 'coverage_evidence', 'alert_time_evidence'):
            with self.assertRaises(ValueError):
                validate_source_contract(dict(self.contract, **{field: 'REQUIRED: confirm'}))
        with self.assertRaises(ValueError):
            validate_source_contract(dict(self.contract, alert_time_semantics='unknown'))

    def test_dictionary_feature_order_exactly_matches_code(self):
        text = Path('docs/FEATURE_DICTIONARY.md').read_text(encoding='utf-8')
        block = text.split('<!-- FEATURE_ORDER_START -->')[1].split('<!-- FEATURE_ORDER_END -->')[0]
        names = block.split('```text\n')[1].split('```')[0].strip().splitlines()
        self.assertEqual(tuple(names), FEATURE_NAMES)

    def test_processed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory)/'processed.sqlite'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE alerts(alert_id TEXT,senior_id TEXT,alert_date TEXT,sos_note TEXT,severity INTEGER)')
            conn.commit(); conn.close()
            with self.assertRaisesRegex(ValueError, 'uncompressed raw'):
                SQLiteSource(db, self.contract)

    def test_conflicting_duplicate_steps_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.features([self.measurement(3, value=10), self.measurement(3, value=11)])

    def test_acquisition_and_delivery_exact_bucket_boundaries(self):
        rows = self.features([self.measurement(5, 'Heartrate', value=80)])
        self.assertIsNone(rows[0]['heartrate'])
        self.assertEqual(rows[1]['heartrate'], 80)
        delayed = self.measurement(3, 'Heartrate', value=80, available_at=iso(self.start+timedelta(minutes=5)))
        rows = self.features([delayed], dict(self.contract, measurement_availability='explicit_available_at'))
        self.assertTrue(all(r['heartrate'] is None for r in rows))

    def test_timezone_dst_and_naive_coverage_fail_closed(self):
        for value in ('2025-03-30T02:30:00', '2025-10-26T02:30:00'):
            with self.assertRaises(ValueError):
                localize(value, dict(self.contract, source_timezone='Europe/Warsaw'))
        with self.assertRaises(ValueError):
            as_time('2030-01-01T00:00:00')
        self.assertEqual(localize('2030-01-01T01:00:00+01:00', {}), '2030-01-01T00:00:00Z')

    def test_config_rejects_consistent_but_different_task(self):
        for changes in ({'grid_minutes': 15, 'sequence_steps': 32}, {'horizon_minutes': 120}, {'hr_sd_minutes': 60}):
            with self.assertRaises(ValueError):
                Task(dict(self.task.values, **changes)).validate()

    def test_future_columns_and_forged_identity(self):
        rows = self.features(self.source.measurements(self.sid))
        stream, _ = generate_patient_manifest(rows, [], self.c, self.task)
        sample = stream[0]
        expected = sequence_for_sample(rows, sample, self.task)
        for row in rows:
            row.update(label_3=1, target=1, future_diagnosis=123)
        self.assertEqual(expected, sequence_for_sample(rows, sample, self.task))
        with self.assertRaises(ValueError):
            sequence_for_sample(rows, dict(sample, sample_id='forged'), self.task)
        bad = copy.deepcopy(rows)
        index = next(i for i,r in enumerate(bad) if r['timestamp'] == sample['input_start'])
        bad[index]['available_at'] = iso(as_time(sample['prediction_time'])+timedelta(minutes=1))
        with self.assertRaises(ValueError):
            sequence_for_sample(bad, sample, self.task)

    def test_last_measurement_is_not_followup(self):
        # All measurements after run-in disappear, but declared ascertainment remains.
        early = [r for r in self.source.measurements(self.sid) if as_time(r['date']) < self.start+timedelta(hours=24)]
        stream, _ = generate_patient_manifest(self.features(early), [], self.c, self.task)
        self.assertEqual(as_time(stream[-1]['prediction_time']), self.start+timedelta(hours=68))
        c = dict(self.c, outcome_coverage_end=iso(self.start+timedelta(hours=40)))
        censored, _ = generate_patient_manifest(self.features(early), [], c, self.task)
        self.assertEqual(as_time(censored[-1]['prediction_time']), self.start+timedelta(hours=36))
        with self.assertRaises(ValueError):
            training_manifest(stream, [], self.task)

    def test_array_reader_rejects_permuted_inputs_even_with_updated_shard_checksum(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            run, arrays = Path(directory)/'run', Path(directory)/'arrays'
            build_run(self.source, self.coverages, self.histories, self.task, self.contract, run, source_snapshot_id='fiction')
            exported = export_sequences(run, arrays, self.task, 'train')
            self.assertGreater(exported['samples'], 0)
            self.assertEqual(sum(len(s['sample_ids']) for s in verified_shards(arrays, self.task)), exported['samples'])
            shard = arrays/exported['shards'][0]['name']
            with np.load(shard, allow_pickle=False) as data:
                payload = {k: data[k] for k in data.files}
            payload['X'][[0, 1]] = payload['X'][[1, 0]]
            np.savez_compressed(shard, **payload)
            exported['shards'][0]['sha256'] = file_hash(shard)
            (arrays/'array_metadata.json').write_text(json.dumps(exported), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'identity/content'):
                list(verified_shards(arrays, self.task))

    def test_sqlite_duckdb_source_parity(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('Install requirements-data.txt and provision DuckDB sqlite extension')
        contract = dict(self.contract)
        extension = os.environ.get('AI_CVD_TEST_EXTENSION_DIRECTORY')
        if extension:
            contract['duckdb_extension_directory'] = extension
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory)/'fiction.sqlite'
            conn = sqlite3.connect(db)
            conn.executescript('CREATE TABLE seniors(id TEXT); CREATE TABLE measurements(id INTEGER,senior_id TEXT,date TIMESTAMP,type TEXT,value REAL,sbp REAL,dbp REAL); CREATE TABLE alerts(alert_id TEXT,senior_id TEXT,alert_date TIMESTAMP,sos_note TEXT);')
            conn.execute('INSERT INTO seniors VALUES (?)', (self.sid,))
            measurements = list(self.source.measurements(self.sid))
            measurements.append(self.measurement(25, 'Heartrate', value='invalid'))
            measurements.append(dict(self.measurement(30, 'Heartrate', value=80), date='2030-01-01T01:30:00+01:00'))
            for i,r in enumerate(measurements):
                conn.execute('INSERT INTO measurements VALUES (?,?,?,?,?,?,?)', (i,self.sid,r['date'],r['type'],r.get('value'),r.get('sbp'),r.get('dbp')))
            for r in self.source.alerts(self.sid):
                conn.execute('INSERT INTO alerts VALUES (?,?,?,?)', (r['alert_id'],self.sid,r['alert_date'],'zdecydowano wezwać ZRM' if r['severity']==3 else 'nawiązano kontakt'))
            conn.commit(); conn.close()
            before = file_hash(db)
            sqlite = SQLiteSource(db, contract)
            other = None
            try:
                other = DuckDBSource(db, contract)
                self.assertEqual(sqlite.patients(), other.patients())
                self.assertEqual(self.features(sqlite.measurements(self.sid)), self.features(other.measurements(self.sid)))
                self.assertEqual(build_episodes(sqlite.alerts(self.sid), self.task), build_episodes(other.alerts(self.sid), self.task))
                outputs = []
                for name, source in [('sqlite',sqlite), ('duckdb',other)]:
                    output = Path(directory)/name
                    build_run(source, [self.c], [self.histories[0]], self.task, contract, output, source_snapshot_id='fiction')
                    verify_run(output, self.task)
                    outputs.append(output)
                for artifact in ('features.jsonl','episodes.jsonl','stream_manifest.jsonl','unsupervised_training_manifest.jsonl','cohort_flow.json'):
                    self.assertEqual((outputs[0]/artifact).read_bytes(), (outputs[1]/artifact).read_bytes(), artifact)
            finally:
                sqlite.close()
                if other is not None:
                    other.close()
            self.assertEqual(file_hash(db), before)


if __name__ == '__main__':
    unittest.main()
