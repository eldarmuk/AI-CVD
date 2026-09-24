import sqlite3
import json
import unittest
import numpy as np
from src.ai_cvd.task import load_task
from src.ai_cvd.synthetic import fixture
from src.ai_cvd.features import build_features, FEATURE_NAMES
from src.ai_cvd.fast_features import read_clean_patient, vector_features
from src.ai_cvd.episodes import build_episodes, event_time, as_time
from src.ai_cvd.dataset import generate_patient_manifest, training_manifest
from src.ai_cvd.compact import window_index
import pandas as pd
import pickle
from datetime import timedelta
import tempfile
from pathlib import Path
from src.ai_cvd.task import Task
from src.ai_cvd.compact import build, sequence_batches
from src.ai_cvd.compact_validate import validate
from src.ai_cvd.compact_report import report
from src.ai_cvd.cli import file_hash
from src.ai_cvd.compact_steps_audit import audit as steps_audit


class RetrospectiveTests(unittest.TestCase):
    def test_compact_synthetic_build_export_and_training_statistics(self):
        task=Task(dict(load_task().values,study_start_local='2030-01-01T00:00:00',study_end_local_exclusive='2030-01-04T00:00:00')).validate()
        source,_,_,_=fixture(task)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);db=root/'fiction.sqlite';c=sqlite3.connect(db)
            c.executescript('CREATE TABLE seniors(id TEXT); CREATE TABLE measurements(senior_id TEXT,date TEXT,type TEXT,value REAL,sbp REAL,dbp REAL); CREATE TABLE alerts(alert_id TEXT,senior_id TEXT,alert_date TEXT,sos_note TEXT);')
            c.executemany('INSERT INTO seniors VALUES (?)',[(sid,) for sid in source.patients()])
            for sid in source.patients():
                for r in source.measurements(sid):
                    c.execute('INSERT INTO measurements VALUES (?,?,?,?,?,?)',(sid,r['date'].replace('T',' ').replace('Z',''),r['type'],r.get('value'),r.get('sbp'),r.get('dbp')))
                for r in source.alerts(sid):
                    note='zdecydowano wezwać ZRM' if r['severity']==3 else ('alarm przypadkowy' if r['severity']==0 else 'nawiązano kontakt')
                    c.execute('INSERT INTO alerts VALUES (?,?,?,?)',(r['alert_id'],sid,r['alert_date'].replace('T',' ').replace('Z',''),note))
            c.commit();c.close()
            contract=root/'contract.toml'
            contract.write_text('''steps_semantics="cumulative_counter"
counter_reset_policy="daily"
steps_evidence="Fictional increments"
measurement_availability="timestamp_is_available_at"
availability_evidence="Fictional immediate availability"
coverage_evidence="Fictional complete alarm capture"
clinical_history_policy="exclude"
source_timezone="UTC"
alert_time_semantics="alarm_initiation_with_retrospective_classification"
alert_time_evidence="Fictional retrospective notes"
[evidence]
outcome_completeness="researcher_attestation"
[provenance]
'''+f'source_db_bytes={db.stat().st_size}\nsource_db_mtime_ns={db.stat().st_mtime_ns}\n')
            run=root/'run';build(db,contract,run,task,workers=1)
            with self.assertRaisesRegex(ValueError,'interrupted builds'):
                build(db,contract,run,task,workers=1,resume=True)
            original={p.name:file_hash(p) for p in (run/'patients').glob('*.npz')}
            saved=root/'previous-finalization';saved.mkdir()
            for p in list((run/'exports').glob('*'))+[run/n for n in ('run_metadata.json','normalization.json','statistics.json','cohort_flow.json')]:
                p.rename(saved/p.name)
            build(db,contract,run,task,workers=1,resume=True)
            self.assertEqual(original,{p.name:file_hash(p) for p in (run/'patients').glob('*.npz')})
            result=validate(run,task)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(steps_audit(run,db,task)['status'],'passed')
            self.assertTrue(report(run).exists())
            for split in ('train','validation','test'):
                batches=list(sequence_batches(run,split,task))
                self.assertEqual(sum(len(b['y']) for b in batches),result['counts'][split]['samples'])
                self.assertTrue(all(b['X'].shape[1:]==(96,len(FEATURE_NAMES)) for b in batches))
                self.assertTrue(all(len(b['episode_ids'])==len(b['sample_ids']) for b in batches))
            # Even refreshed file checksums cannot disguise permuted immutable IDs.
            meta=json.loads((run/'run_metadata.json').read_text())
            shard=next(s for s in meta['patient_shards'] if s['split']=='train')
            path=run/shard['file']
            with np.load(path) as z:
                content={key:z[key] for key in z.files}
            content['sample_ids']=content['sample_ids'][::-1]
            np.savez_compressed(path,**content)
            digest=file_hash(path);shard['sha256']=digest
            meta['artifacts_sha256'][shard['file']]=digest
            for export_path in (run/'exports').glob('*.json'):
                export=json.loads(export_path.read_text())
                for s in export['shards']:
                    if s['file']==shard['file']:
                        s['sha256']=digest
                export_path.write_text(json.dumps(export))
                meta['artifacts_sha256']['exports/'+export_path.name]=file_hash(export_path)
            (run/'run_metadata.json').write_text(json.dumps(meta))
            result['run_metadata_sha256']=file_hash(run/'run_metadata.json')
            (run/'integrity_verification.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'Immutable sample alignment'):
                next(sequence_batches(run,'train',task))

    def test_task_worker_serialization(self):
        task=load_task()
        self.assertEqual(pickle.loads(pickle.dumps(task)).identifier,task.identifier)

    def test_observed_support_is_not_enrollment(self):
        task=load_task();source,coverage,_,contract=fixture(task)
        c=dict(coverage[0]);c['support_start']=c.pop('enrollment_time')
        c['coverage_basis']='researcher_attested_complete_alerts_with_observed_support'
        rows=list(build_features(source.measurements(c['senior_id']),c,[],task,contract))
        manifest,_=generate_patient_manifest(rows,[],c,task)
        self.assertGreater(len(manifest),0)
        self.assertNotIn('enrollment_time',c)
        self.assertTrue(all(as_time(r['prediction_time'])+timedelta(minutes=240)<=as_time(c['outcome_coverage_end']) for r in manifest))

    def test_compact_window_index_matches_reference(self):
        task=load_task()
        source,coverage,_,contract=fixture(task)
        for c in coverage:
            rows=list(build_features(source.measurements(c['senior_id']),c,[],task,dict(contract,clinical_history_policy='exclude')))
            frame=pd.DataFrame([{f:r[f] for f in FEATURE_NAMES} for r in rows],index=pd.to_datetime([r['timestamp'] for r in rows],utc=True)).astype(float)
            frame.index=frame.index.as_unit('ns')
            episodes=build_episodes(source.alerts(c['senior_id']),task)
            reference,flow=generate_patient_manifest(rows,episodes,c,task)
            actual,fastflow,events=window_index(frame,episodes,c,task)
            self.assertEqual(actual['target'].tolist(),[r['target'] for r in reference])
            np.testing.assert_allclose(actual['lead_minutes'],np.array([r['lead_minutes'] for r in reference],dtype=float),equal_nan=True)
            for key in set(flow)|set(fastflow):
                self.assertEqual(fastflow.get(key,0),flow.get(key,0),key)
            train=training_manifest([dict(r,split='train') for r in reference],episodes,task,coverage=c)
            self.assertEqual(int(actual['training'].sum()),len(train))

    def test_vectorized_feature_parity(self):
        task = load_task()
        source, coverages, _, contract = fixture(task)
        contract = dict(contract, clinical_history_policy='exclude', steps_semantics='cumulative_counter', counter_reset_policy='daily',
                        alert_time_semantics='alarm_initiation_with_retrospective_classification')
        c = sqlite3.connect(':memory:')
        self.addCleanup(c.close)
        c.execute('CREATE TABLE measurements(senior_id TEXT,date TEXT,type TEXT,value REAL,sbp REAL,dbp REAL)')
        readings = list(source.measurements(coverages[0]['senior_id']))
        readings += [dict(readings[0], type='BloodPressure',value=None,sbp=120,dbp=None),
                     dict(readings[0], type='BloodPressure',value=None,sbp=None,dbp=80)]
        c.executemany('INSERT INTO measurements VALUES (?,?,?,?,?,?)',[(r['senior_id'],r['date'],r['type'],r.get('value'),r.get('sbp'),r.get('dbp')) for r in readings])
        clean = read_clean_patient(c,coverages[0]['senior_id'],task,'2029','2031')
        actual = vector_features(clean,coverages[0],task,contract).to_numpy()
        reference = list(build_features(readings,coverages[0],[],task,contract))
        expected = np.array([[r[f] for f in FEATURE_NAMES] for r in reference],dtype=float)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7,equal_nan=True)

    def test_retrospective_severity_does_not_claim_decision_time(self):
        task = load_task()
        alerts = [{'senior_id':'fiction','alert_id':'a','alert_date':'2030-01-01T10:00:00Z','severity':1},
                  {'senior_id':'fiction','alert_id':'b','alert_date':'2030-01-01T10:05:00Z','severity':3,'note_available_at':'2030-01-02T00:00:00Z'}]
        e = build_episodes(alerts,task)[0]
        self.assertEqual(event_time(e,[3]),as_time(alerts[1]['alert_date']))
        self.assertIsNone(e['escalation_recorded_at'])
        self.assertEqual(e['maximum_severity'],3)
