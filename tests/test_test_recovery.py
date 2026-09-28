"""Fictional interrupted evaluation; never uses private patient shards."""
import tempfile
import unittest
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts import test_evaluation as ev
from scripts import test_recovery as recovery


class NullRisk(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.anchor=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,view):
        assert not torch.is_grad_enabled()
        return torch.zeros(len(view.values))


@contextmanager
def fixture():
    _,_,task=ev.load()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root=Path(directory); (root/'exports').mkdir(); (root/'patients').mkdir(); (root/'audit').mkdir()
        shards=[]; ends=np.arange(95,101); times=(ends+1)*task.grid_minutes*60*10**9
        for i in range(3):
            patient=f'fiction-{i}'
            ids=[ev.sample_id(patient,datetime.fromtimestamp(int(t)/1e9,timezone.utc)) for t in times]
            x=np.full((101,len(ev.FEATURE_NAMES)),np.nan,dtype=np.float32)
            for name in ev.FEATURE_NAMES:
                if name.startswith('observed_') or name in ('hour_sin','hour_cos','is_night','steps_counter_reset'): x[:,ev.FEATURE_NAMES.index(name)]=0
            file=f'patients/{i}.npz'; first=np.searchsorted([times[2]],times,side='right'); stop=np.ones(6,dtype=int)
            np.savez(root/file,features=x,end_rows=ends,grid_start_ns=np.array(0),
                sample_ids=np.array([list(bytes.fromhex(s[7:])) for s in ids],dtype=np.uint8),
                target=stop>first,first_event=first,event_stop=stop)
            shards.append({'senior_id':patient,'file':file,'samples':6,'split':'test',
                'sha256':ev.digest(root/file),'events':[[int(times[2]),f'event-{i}']]})
        ev.write_json(root/'exports/test-stream.json',{'shards':shards})
        receipt={'windows':18,'test_export_sha256':ev.digest(root/'exports/test-stream.json')}
        ev.write_json(root/'receipt.json',receipt); ev.write_json(root/'engineering.json',{})
        ev.write_json(root/'audit/validation_results.json',{'discrimination':{},'whole_validation_operational_descriptive':{}})
        e={'threshold':.4,'calibration':{'intercept':-.2},'alert_policy':{'cooldown_minutes':240},'metric_definitions':{}}
        blind={'shards':[{k:v for k,v in s.items() if k!='events'} for s in shards]}
        bundle=(receipt,e,task,NullRisk(),{'mean':[0]*6,'scale':[1]*6},root,blind)
        stack.enter_context(patch.object(ev,'OUTPUT',root/'out')); stack.enter_context(patch.object(ev,'AUDIT',root/'audit'))
        stack.enter_context(patch.object(ev,'ENGINEERING',root/'engineering.json'))
        stack.enter_context(patch.object(ev,'preflight',return_value=bundle))
        stack.enter_context(patch.object(ev.subprocess,'check_output',return_value=''))
        scorer=ev.score_patient; calls=[]
        def interrupt(*args):
            calls.append(True)
            if len(calls)==2: raise RuntimeError('Fictional accidental interruption')
            return scorer(*args)
        with patch.object(ev,'score_patient',side_effect=interrupt):
            try: ev.execute(root/'receipt.json')
            except RuntimeError as error:
                assert str(error)=='Fictional accidental interruption'
        yield root,bundle


class RecoveryTests(unittest.TestCase):
    def test_continuation_preserves_prefix_and_matches_uninterrupted(self):
        with fixture() as (root,bundle):
            receipt,e,task,model,scaler,_,blind=bundle; out=ev.OUTPUT
            rows,pos,snapshot=recovery.verify_prefix(out,root,blind,task,e,18)
            self.assertEqual((len(rows),pos),(1,6))
            old={name:ev.digest(out/name) for name in ('started.json','patient-000000.json','patient-000000.npz')}
            prefix={k:np.load(out/f'{k}.npy')[:6].copy() for k in recovery.TYPES}
            report={'snapshot_sha256':snapshot,'seconds':0}
            ev.write_json(root/'recovery-audit.json',report)
            with patch.object(recovery,'audit',return_value=(report,bundle,receipt,rows)), \
                 patch.object(ev,'score_patient',wraps=ev.score_patient) as scorer:
                ev.execute(root/'receipt.json',recovery_audit=root/'recovery-audit.json')
                self.assertEqual(scorer.call_count,2)
            ev.verify_seal(out)
            for name,h in old.items(): self.assertEqual(ev.digest(out/name),h)
            for key,value in prefix.items(): np.testing.assert_array_equal(np.load(out/f'{key}.npy')[:6],value)
            baseline=root/'baseline'
            with patch.object(ev,'OUTPUT',baseline): ev.execute(root/'receipt.json')
            for key in recovery.TYPES: np.testing.assert_array_equal(np.load(out/f'{key}.npy'),np.load(baseline/f'{key}.npy'))
            resumed=ev.read(out/'test_results.json'); original=ev.read(baseline/'test_results.json')
            self.assertEqual(resumed['discrimination'],original['discrimination'])
            self.assertEqual(resumed['operational'],original['operational'])
            with self.assertRaises(ValueError): recovery.verify_prefix(out,root,blind,task,e,18)

    def test_partial_chunk_and_tampering_rejected(self):
        with fixture() as (root,bundle):
            _,e,task,_,_,_,blind=bundle; out=ev.OUTPUT
            path=out/'patient-000001.npz'; path.write_bytes(b'partial')
            with self.assertRaises(ValueError): recovery.verify_prefix(out,root,blind,task,e,18)
            path.unlink()  # disposable fictional fixture only
            with open(out/'patient-000000.npz','ab') as f: f.write(b'tampered')
            with self.assertRaises(ValueError): recovery.verify_prefix(out,root,blind,task,e,18)

    def test_cache_prefix_and_tail_rejected(self):
        with fixture() as (root,bundle):
            _,e,task,_,_,_,blind=bundle; out=ev.OUTPUT
            a=np.load(out/'scores.npy',mmap_mode='r+'); original=float(a[0]); a[0]=.9; a.flush()
            with self.assertRaises(AssertionError): recovery.verify_prefix(out,root,blind,task,e,18)
            a[0]=original; a[-1]=.9; a.flush()
            with self.assertRaises(ValueError): recovery.verify_prefix(out,root,blind,task,e,18)
            a._mmap.close()

    def test_stale_audit_fails_before_provenance_write(self):
        with fixture() as (root,bundle):
            ev.write_json(root/'audit.json',{'snapshot_sha256':{'wrong':'hash'},'seconds':0})
            with patch.object(recovery,'audit',return_value=({'snapshot_sha256':{},'seconds':1},bundle,{},[])):
                with self.assertRaises(ValueError): recovery.prepare(root/'receipt.json',root/'audit.json')
            self.assertEqual(list(ev.OUTPUT.glob('recovery-*.json')),[])

    def test_second_interruption_reaudits_longer_prefix(self):
        with fixture() as (root,bundle):
            receipt,e,task,_,_,_,blind=bundle; out=ev.OUTPUT
            rows,_,snapshot=recovery.verify_prefix(out,root,blind,task,e,18)
            report={'snapshot_sha256':snapshot,'seconds':0}; ev.write_json(root/'audit-1.json',report)
            scorer=ev.score_patient; calls=[]
            def interrupt(*args):
                calls.append(1)
                if len(calls)==2: raise RuntimeError('Second accidental interruption')
                return scorer(*args)
            with patch.object(recovery,'audit',return_value=(report,bundle,receipt,rows)), patch.object(ev,'score_patient',side_effect=interrupt):
                with self.assertRaises(RuntimeError): ev.execute(root/'receipt.json',recovery_audit=root/'audit-1.json')
            rows,pos,snapshot=recovery.verify_prefix(out,root,blind,task,e,18)
            self.assertEqual((len(rows),pos),(2,12)); saved=ev.digest(out/'patient-000001.npz')
            report={'snapshot_sha256':snapshot,'seconds':0}; ev.write_json(root/'audit-2.json',report)
            with patch.object(recovery,'audit',return_value=(report,bundle,receipt,rows)), patch.object(ev,'score_patient',wraps=scorer) as spy:
                ev.execute(root/'receipt.json',recovery_audit=root/'audit-2.json')
                self.assertEqual(spy.call_count,1)
            self.assertEqual(ev.digest(out/'patient-000001.npz'),saved)
            self.assertTrue((out/'recovery-0002.json').exists()); ev.verify_seal(out)

    def test_mutex_rejects_concurrent_entry(self):
        with recovery.exclusive_recovery():
            with self.assertRaises(RuntimeError):
                with recovery.exclusive_recovery(): pass


if __name__=='__main__': unittest.main()
