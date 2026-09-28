import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from contextlib import ExitStack
from datetime import datetime, timezone
from scripts import test_evaluation as ev
from scripts.validation_audit_math import fast_alerts, accumulator, add, summary, stream_metrics
from src.final_study.policy import patient_alerts, calibrated
from scripts.test_uncertainty import weighted_ranking, binomial_interval, interval


class OneShotTests(unittest.TestCase):
    def test_fictional_complete_run_reload_and_one_shot(self):
        class InferenceOnly(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.anchor=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
            def forward(self,view):
                if torch.is_grad_enabled(): raise AssertionError('Gradients enabled')
                return torch.zeros(len(view.values))
        _,_,task=ev.load()
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root=Path(d); (root/'exports').mkdir(); (root/'patients').mkdir(); audit=root/'audit'; audit.mkdir()
            out=root/'out'; ends=np.arange(95,101); times=(ends+1)*task.grid_minutes*60*10**9
            ids=[ev.sample_id('fictional',datetime.fromtimestamp(int(t)/1e9,timezone.utc)) for t in times]
            events=[[int(times[2]),'fictional-event']]
            first=np.searchsorted([times[2]],times,side='right'); stop=np.ones(6,dtype=int)
            x=np.full((101,len(ev.FEATURE_NAMES)),np.nan,dtype=np.float32)
            x[:,ev.FEATURE_NAMES.index('hour_sin')]=0; x[:,ev.FEATURE_NAMES.index('hour_cos')]=1
            x[:,ev.FEATURE_NAMES.index('is_night')]=0
            for f in ev.FEATURE_NAMES:
                if f.startswith('observed_') or f=='steps_counter_reset': x[:,ev.FEATURE_NAMES.index(f)]=0
            np.savez(root/'patients/fiction.npz',features=x,end_rows=ends,grid_start_ns=np.array(0),
                sample_ids=np.array([list(bytes.fromhex(i[7:])) for i in ids],dtype=np.uint8),
                target=stop>first,first_event=first,event_stop=stop)
            shard={'senior_id':'fictional','file':'patients/fiction.npz','samples':6,'split':'test',
                'sha256':ev.digest(root/'patients/fiction.npz'),'events':events}
            ev.write_json(root/'exports/test-stream.json',{'shards':[shard]})
            receipt={'windows':6,'test_export_sha256':ev.digest(root/'exports/test-stream.json')}
            ev.write_json(root/'receipt.json',receipt); ev.write_json(root/'engineering.json',{})
            ev.write_json(audit/'validation_results.json',{'discrimination':{},'whole_validation_operational_descriptive':{}})
            e={'threshold':.4,'calibration':{'intercept':-.2},'alert_policy':{'cooldown_minutes':240},'metric_definitions':{}}
            blind={'shards':[{k:v for k,v in shard.items() if k!='events'}]}
            stack.enter_context(patch.object(ev,'OUTPUT',out)); stack.enter_context(patch.object(ev,'AUDIT',audit))
            stack.enter_context(patch.object(ev,'ENGINEERING',root/'engineering.json'))
            stack.enter_context(patch.object(ev,'preflight',return_value=(receipt,e,task,InferenceOnly(),{'mean':[0]*6,'scale':[1]*6},root,blind)))
            stack.enter_context(patch.object(ev.subprocess,'check_output',return_value=''))
            stack.enter_context(patch.object(torch.optim,'Adam',side_effect=AssertionError('Optimizer forbidden')))
            ev.execute(root/'receipt.json'); ev.verify_seal(out)
            result=ev.read(out/'test_results.json')
            self.assertEqual(result['operational']['all']['detected_episodes'],1)
            self.assertEqual(result['operational']['physiology_observed']['patients'],0)
            self.assertIsNone(result['operational']['physiology_observed']['episode_sensitivity'])
            with np.load(out/'patient-000000.npz',allow_pickle=False) as z:
                self.assertEqual(z['sample_ids'].tolist(),ids)
                np.testing.assert_allclose(z['calibrated_scores'],calibrated(z['scores'],-.2))
            from scripts import test_uncertainty as uncertainty
            with patch.object(uncertainty,'OUTPUT',out), patch.object(uncertainty,'REPLICATES',5):
                uncertainty.run(); ev.verify_seal(out/'uncertainty-v1')
                self.assertEqual(ev.read(out/'uncertainty-v1/uncertainty_results.json')['replicates'],5)
                with self.assertRaises(FileExistsError): uncertainty.run()
            with self.assertRaises(FileExistsError): ev.execute(root/'receipt.json')

    def test_metadata_skip_outcomes_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x.json'
            p.write_text('{"shards":[{"senior_id":"x","events":[[123,"a\\\"b"]],"target":1,"samples":4}]}')
            self.assertEqual(ev.selected_json(p,{'shards':{'senior_id':None,'samples':None}}),
                             {'shards':[{'senior_id':'x','samples':4}]})
            p.write_text('{"samples":1,"samples":2}')
            with self.assertRaises(ValueError): ev.selected_json(p,{'samples':None})

    def test_tamper_and_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root/'artifact'; p.write_text('frozen'); h = ev.digest(p)
            ev.check_hash(p,h); p.write_text('changed')
            with self.assertRaises(ValueError): ev.check_hash(p,h)
            out = root/'out'
            with patch.object(ev,'OUTPUT',out):
                ev.reserve(out)
                with self.assertRaises(FileExistsError): ev.reserve(out)
                with self.assertRaises(ValueError): ev.reserve(root/'alternative')

    def test_no_fitting_or_gradient_calls(self):
        forbidden = {'backward','grad','enable_grad','set_grad_enabled','fit_intercept','select_threshold',
                     'fit_scaler','fit','train','Adam','SGD','LBFGS','minimize'}
        for name in ('scripts/test_evaluation.py','scripts/test_uncertainty.py','scripts/test_recovery.py'):
            tree = ast.parse((ev.ROOT/name).read_text())
            calls = {n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
                     for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
            self.assertFalse(calls & forbidden)
        self.assertIn('@torch.inference_mode()', (ev.ROOT/'scripts/test_evaluation.py').read_text())

    def test_failed_preflight_never_reads_export_or_reserves(self):
        with patch.object(ev,'OUTPUT',Path('nonexistent-one-shot-synthetic')), \
             patch.object(ev,'preflight',side_effect=ValueError('binding')), \
             patch.object(ev,'reserve') as reserve, patch.object(ev,'read') as read:
            with self.assertRaises(ValueError): ev.execute(Path('unused'))
            reserve.assert_not_called(); read.assert_not_called()

    def test_policy_boundaries_cooldown_matching_and_denominator(self):
        task = SimpleNamespace(grid_minutes=5,horizon_minutes=240)
        minute = 60*10**9
        times = np.arange(0,486,5,dtype=np.int64)*minute
        p = np.zeros(len(times)); p[[0,2,48,50,96]] = .8
        events = [[0,'at-cutoff'],[240*minute,'upper'],[241*minute,'next'],[480*minute,'last']]
        vital = np.arange(len(times))%2==0
        actual = fast_alerts(times,p,events,.8,task,240,vital)
        self.assertEqual(actual,patient_alerts(times,p,events,.8,task,240,vital))
        self.assertEqual([a['prediction_ns']//minute for a in actual['alerts']],[0,240,480])
        self.assertEqual([a['episode_id'] for a in actual['alerts']],['upper','next',None])
        self.assertEqual([a['lead_minutes'] for a in actual['alerts']],[240.,1.,None])
        a = accumulator(); add(a,actual); s = summary(a,task)
        self.assertEqual(s['eligible_episodes'],3); self.assertEqual(s['episode_sensitivity'],2/3)
        self.assertEqual(s['supported_patient_days'],len(times)*5/1440)
        self.assertEqual(s['false_alerts_per_supported_day'],1/s['supported_patient_days'])
        sub = ev.subgroup_result(actual,times,vital,events,task)
        self.assertEqual(sub['alerts'],actual['alerts'])
        # Intercept used as supplied, with no estimation.
        np.testing.assert_allclose(calibrated(np.array([.5]),-np.log(3)),[.25])

    def test_persistent_crossing_gap_and_duplicate_episode(self):
        task = SimpleNamespace(grid_minutes=5,horizon_minutes=240); m=60*10**9
        times = np.array([0,5,240,245,480])*m; p = np.ones(5)
        result = fast_alerts(times,p,[[240*m,'event']],.5,task,240,np.ones(5,dtype=bool))
        self.assertEqual(len(result['alerts']),3)
        self.assertEqual(len(result['matched_episodes']),1)
        with self.assertRaises(ValueError): patient_alerts(times,p,[[1,'x'],[2,'x']],.5,task,240)

    def test_bootstrap_matches_duplicated_patients_and_ties(self):
        p = np.array([.8,.8,.3,.1]); y = np.array([1,0,1,0],dtype=np.uint8); pid=np.array([0,0,1,1])
        w = np.array([2,1]); auc,ap = weighted_ranking(y,pid,np.array([1,2,3]),w)
        idx = np.repeat(np.arange(4),w[pid]); reference=stream_metrics(y[idx],p[idx],0)
        self.assertAlmostEqual(auc,reference['auroc']); self.assertAlmostEqual(ap,reference['ap'])
        self.assertEqual(binomial_interval(0,0),None)
        self.assertEqual(binomial_interval(0,10)[0],0.)
        self.assertEqual(interval([None,1,0])['undefined_replicates'],1)

    def test_evidence_empty_zero_steps_and_single_value(self):
        _,_,task=ev.load(); x=np.full((100,len(ev.FEATURE_NAMES)),np.nan,dtype=np.float32)
        x[99,ev.FEATURE_NAMES.index('steps')]=0
        vital,primitive=ev.evidence(x,np.array([95,99]),task)
        self.assertEqual(vital.tolist(),[False,False]); self.assertEqual(primitive.tolist(),[False,True])
        x[95,ev.FEATURE_NAMES.index('heartrate')]=60
        self.assertTrue(ev.evidence(x,np.array([95]),task)[0][0])


if __name__ == '__main__': unittest.main()
