import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.final_study.config import load,seal,verify_seal
from src.final_study.policy import (patient_alerts,evaluate,fit_intercept,calibrated,
    authorize_validation,select_threshold,calibration_partition)
from src.architecture_study.config import Scope,fingerprint
from src.architecture_study.data import digest
from src.ai_cvd.dataset import patient_split


class FinalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.spec,self.c,self.task = load()
        self.minute = 60*10**9

    def test_r0_projection_and_budget_frozen(self):
        self.assertEqual(self.c['recipes'],['R0'])
        self.assertEqual(self.c['projection'],'primitive6_process25_v1')
        self.assertEqual(self.c['resources']['ssl_updates'],4000)
        self.assertEqual(self.task.grid_minutes*self.task.sequence_steps,self.task.lookback_minutes)

    def test_alert_horizon_open_closed_one_to_one_and_no_point_adjustment(self):
        t = np.array([0,5,10])*self.minute
        result = patient_alerts(t,[.9,.1,.9],[(0,'at_cutoff'),(240*self.minute,'boundary')],.5,self.task,5)
        self.assertEqual([a['episode_id'] for a in result['alerts']],['boundary',None])
        self.assertEqual(result['matched_episodes'],['boundary'])
        self.assertEqual(len(result['alerts']),2)

    def test_cooldown_exact_boundary_and_sustained_high(self):
        t = np.array([0,5,235,240,245,480])*self.minute
        r = patient_alerts(t,[1,0,1,0,1,1],[],.5,self.task,240)
        self.assertEqual([a['prediction_ns']//self.minute for a in r['alerts']],[0,245])
        r = patient_alerts(np.array([0,5,240])*self.minute,[1,0,1],[],.5,self.task,240)
        self.assertEqual(len(r['alerts']),2)
        r = patient_alerts(np.arange(100)*5*self.minute,np.ones(100),[],.5,self.task,240)
        self.assertEqual(len(r['alerts']),1)

    def test_episode_metrics_supported_days_and_evidence(self):
        stream = {'a':{'times':np.array([0,5,10])*self.minute,'scores':[.9,.1,.9],
                       'vital':[True,False,False],'events':[(5*self.minute,'e')]}}
        r = evaluate(stream,.5,self.task,5)
        self.assertEqual(r['episode_sensitivity'],1)
        self.assertEqual(r['alert_precision'],.5)
        self.assertEqual(r['false_alerts'],1)
        self.assertAlmostEqual(r['supported_patient_days'],15/1440)
        self.assertEqual(r['lead_minutes_quantiles'],[5.]*5)
        self.assertEqual(r['evidence_strata']['True']['matched_alerts'],1)

    def test_validation_guards_and_intercept(self):
        ids = {split:next(f'fake-{i}' for i in range(1000) if patient_split(f'fake-{i}',self.task)==split)
               for split in ('train','validation','test')}
        for split in ('train','test'):
            with self.assertRaises(ValueError):
                fit_intercept([.1,.1],[0,1],split=split,patients=[ids[split]]*2,task=self.task)
            with self.assertRaises(ValueError):
                authorize_validation('validation',[ids[split]],self.task)
            with self.assertRaises(ValueError):
                select_threshold({ids[split]:{}},0,split=split,task=self.task,config=self.spec['validation'])
        fitted = fit_intercept([.1]*10,[1]+[0]*9,split='validation',patients=[ids['validation']]*10,task=self.task)
        self.assertAlmostEqual(float(calibrated([.1],fitted['intercept'])[0]),.1)
        fallback = fit_intercept([.1],[0],split='validation',patients=[ids['validation']],task=self.task)
        self.assertEqual(fallback['status'],'identity_insufficient_classes')
        self.assertEqual(calibration_partition(ids['validation'],91027),calibration_partition(ids['validation'],91027))

    def test_identity_and_scope_reject_nonfit_population(self):
        s = Scope(frozenset(['a','b']),frozenset(['a']),frozenset(['b']),0)
        for operation in ('ssl','scaler','head'):
            with self.assertRaises(ValueError): s.authorize(['b'],operation)
            with self.assertRaises(ValueError): s.authorize(['a'],operation,split='validation')
            with self.assertRaises(ValueError): s.authorize(['a'],operation,split='test')

    def test_integrity_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'x').write_text('correct'); seal(root)
            verify_seal(root); (root/'x').write_text('changed')
            with self.assertRaises(ValueError): verify_seal(root)

    def test_fictional_training_reload_and_completed_skip(self):
        from test_architecture_study import fixture
        from src.final_study.training import train,verify_run
        from src.architecture_study.data import TrainingStore
        from src.architecture_study.projection import project
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); store,scopes,_,task = fixture(root)
            # Entirely fictional validation export, installed before freezing the run.
            from datetime import datetime,timezone
            from src.ai_cvd.dataset import sample_id
            from src.ai_cvd.features import FEATURE_NAMES
            from src.architecture_study.data import write_json
            val_ids=[]
            for bucket in (True,False):
                val_ids.extend([f'val-{i}' for i in range(1000) if patient_split(f'val-{i}',task)=='validation'
                                and calibration_partition(f'val-{i}',91027)==bucket][:2])
            shards=[]
            for patient in val_ids:
                with np.load(root/store.shards[0]['file'],allow_pickle=False) as z:
                    a={k:z[k] for k in z.files}
                times=int(a['grid_start_ns'])+(a['end_rows'].astype(np.int64)+1)*task.grid_minutes*60*10**9
                a['sample_ids']=np.array([list(bytes.fromhex(sample_id(patient,datetime.fromtimestamp(int(t)/1e9,timezone.utc)).removeprefix('sample_'))) for t in times],dtype=np.uint8)
                file=f'patients/{patient}.npz'; np.savez_compressed(root/file,**a)
                shards.append({'senior_id':patient,'split':'validation','file':file,'sha256':digest(root/file),
                               'samples':len(times),'events':store.shards[0]['events']})
            write_json(root/'exports/validation-stream.json',{'shards':shards,'task_identifier':task.identifier,'feature_names':FEATURE_NAMES})
            meta=json.loads((root/'run_metadata.json').read_text())
            meta['artifacts_sha256']['exports/validation-stream.json']=digest(root/'exports/validation-stream.json')
            (root/'run_metadata.json').write_text(json.dumps(meta))
            (root/'integrity_verification.json').write_text(json.dumps({'status':'passed','run_metadata_sha256':digest(root/'run_metadata.json')}))
            store=TrainingStore(root,task)
            spec,c = copy.deepcopy(self.spec),copy.deepcopy(self.c)
            spec.update(canonical_run=str(root),run_metadata_sha256=store.run_hash,train_export_sha256=store.export_hash,
                        checkpoint_partition_count=3,ssl_draws=2,checkpoint_every=1,checkpoint_batches=1)
            c['canonical_run'] = str(root)
            c['resources'].update(ssl_updates=2,checkpoint_every=1,checkpoint_batches=1,scaler_windows=4,batch_size=4)
            c['sampling']['patients_per_batch']=2
            output = root/'final'
            train(spec,c,task,output)
            model,scaler = verify_run(output,spec,c)
            from src.architecture_study.synthetic import tensors
            v = project(tensors(2),scaler['mean'],scaler['scale'],task)
            with torch.no_grad():
                a = model(v); b = model(v)
                self.assertTrue(torch.equal(a,b))
                reloaded,_ = verify_run(output,spec,c)
                self.assertTrue(torch.equal(a,reloaded(v)))
            hashes = {p.name:digest(p) for p in output.iterdir()}
            train(spec,c,task,output,resume=True)
            self.assertEqual(hashes,{p.name:digest(p) for p in output.iterdir()})
            changed = copy.deepcopy(c); changed['projection']='wrong'
            with self.assertRaises(ValueError): verify_run(output,spec,changed)
            from src.final_study.validation import score,develop
            predictions=root/'validation'; development=root/'development'
            score(spec,c,task,output,predictions)
            develop(spec,c,task,output,predictions,development)
            verify_seal(predictions); verify_seal(development)
            prov=json.loads((development/'provenance.json').read_text())
            self.assertFalse(set(prov['calibration_patients']) & set(prov['policy_patients']))


if __name__ == '__main__':
    unittest.main()
