"""Fictional end-to-end audit/freeze, without training any model."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone
import numpy as np
from scripts import audit_validation as av
from src.final_study.config import load
from src.final_study.policy import calibration_partition,fit_intercept,select_threshold,evaluate,calibrated
from src.architecture_study.analyze import metrics
from src.architecture_study.data import write_json,digest
from src.ai_cvd.dataset import patient_split,sample_id
from src.ai_cvd.features import FEATURE_NAMES


class Integration(unittest.TestCase):
    def test_fictional_audit_and_freeze(self):
        spec,c,task=load()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; source.mkdir(); (source/'patients').mkdir()
            scores=root/'scores'; scores.mkdir(); policy=root/'policy'; policy.mkdir(); model=root/'model'; model.mkdir()
            for name in ('complete.json','R0-ssl.pt','R0-risk.pt','head_fit.json','scaler.json'):
                (model/name).write_text('{}')
            write_json(model/'provenance.json',{'source_sha256':{}})
            patients=[next(f'fake-val-{i}' for i in range(10000) if patient_split(f'fake-val-{i}',task)=='validation'
                and calibration_partition(f'fake-val-{i}',spec['validation']['seed'])==flag) for flag in (True,False)]
            manifest=[]; shards=[]; streams={}; origin=1761955200000000000
            for i,p in enumerate(patients):
                features=np.full((200,len(FEATURE_NAMES)),np.nan,dtype=np.float32)
                features[::3,FEATURE_NAMES.index('heartrate')]=70
                ends=np.arange(task.sequence_steps-1,200); times=origin+(ends+1)*task.grid_minutes*60*10**9
                events=[(int(times[60]),f'event-{i}')]
                y=((times<events[0][0]) & (times+task.horizon_minutes*60*10**9>=events[0][0])).astype(np.uint8)
                ids=[sample_id(p,datetime.fromtimestamp(int(t)/1e9,timezone.utc)) for t in times]
                native=np.array([list(bytes.fromhex(s[7:])) for s in ids],dtype=np.uint8)
                file=f'patients/{i}.npz'
                np.savez_compressed(source/file,features=features,end_rows=ends,grid_start_ns=origin,sample_ids=native,target=y)
                predfile=f'p{i}.npz'; ps=np.linspace(.01,.3,len(times)); vital=np.ones(len(times),dtype=bool)
                np.savez_compressed(scores/predfile,sample_ids=np.array(ids),times=times,scores=ps,vital=vital,labels=y)
                manifest.append({'patient':p,'file':predfile,'samples':len(times),'events':events,'sha256':digest(scores/predfile)})
                shards.append({'file':file,'sha256':digest(source/file),'events':events})
                streams[p]={'times':times,'scores':ps,'labels':y,'vital':vital,'events':events}
            d=streams[patients[0]]
            fit=fit_intercept(d['scores'],d['labels'],split='validation',patients=[patients[0]]*len(d['labels']),task=task)
            part={patients[1]:streams[patients[1]]}
            threshold=select_threshold(part,fit['intercept'],split='validation',task=task,config=spec['validation'])
            record=evaluate(part,threshold['selected']['threshold'],task,spec['validation']['cooldown_minutes'],fit['intercept'])
            d=streams[patients[1]]; record['window_diagnostics']={'all':{
                'raw':metrics(d['labels'],d['scores'],np.ones(len(d['labels']))),
                'calibrated':metrics(d['labels'],calibrated(d['scores'],fit['intercept']),np.ones(len(d['labels'])))}}
            write_json(policy/'calibration.json',fit); write_json(policy/'threshold.json',threshold)
            write_json(policy/'policy_development_metrics.json',record)
            (policy/'complete.json').write_text('{}'); (scores/'complete.json').write_text('{}')
            returned=(spec,c,task,None,None,source,manifest,{'shards':shards},[patients[0]],[patients[1]])
            with patch.object(av,'ROOT',root),patch.object(av,'MODEL',model),patch.object(av,'SCORES',scores),patch.object(av,'POLICY',policy),patch.object(av,'metadata',return_value=returned),patch.object(av.subprocess,'check_output',return_value='fictional-git\n'):
                av.audit(root/'runs/audit')
            self.assertTrue((root/'runs/audit/final_evaluation_spec.json').exists())
            self.assertTrue((root/'runs/audit/complete.json').exists())


if __name__=='__main__': unittest.main()
