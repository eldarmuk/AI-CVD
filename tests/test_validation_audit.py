import unittest
import numpy as np
from scripts.validation_audit_math import fast_alerts,stream_metrics
from scripts.validation_support import vector_windows
from src.final_study.config import load
from src.final_study.policy import patient_alerts,calibrated
from src.architecture_study.analyze import metrics


class ValidationAuditTests(unittest.TestCase):
    def test_vector_prefix_exact_and_no_future(self):
        x=np.arange(200*3).reshape(200,3); ends=np.array([95,99,120])
        expected=np.stack([x[e-95:e+1] for e in ends])
        np.testing.assert_array_equal(vector_windows(x,ends,96),expected)
        x[121:]=-999
        np.testing.assert_array_equal(vector_windows(x,ends,96),expected)
        with self.assertRaises(ValueError): vector_windows(x,[94],96)

    def test_vector_policy_exact_random_gaps_ties_and_boundaries(self):
        _,_,task=load(); rng=np.random.default_rng(31)
        for _ in range(20):
            times=np.cumsum(rng.choice([5,5,5,10,250],size=100)).astype(np.int64)*60*10**9
            scores=rng.choice([.1,.5,.9],size=100); vital=rng.integers(2,size=100).astype(bool)
            events=[(int(times[10]),'same_time'),(int(times[30]+task.horizon_minutes*60e9),'exact_horizon'),(int(times[70]),'later')]
            for threshold in (.1,.5,.9,1.1):
                self.assertEqual(fast_alerts(times,scores,events,threshold,task,240,vital),
                                 patient_alerts(times,scores,events,threshold,task,240,vital))

    def test_stream_metrics_match_reference_and_calibration_ties(self):
        y=np.array([0,1,0,0,1],dtype=np.uint8); p=np.array([.1,.1,.8,0.,1.])
        result=stream_metrics(y,p,-.2)
        for name,scores in [('raw',p),('calibrated',calibrated(p,-.2))]:
            reference=metrics(y,scores,np.ones(len(y)))
            for k in ('mean_risk','log_loss','brier'):
                self.assertAlmostEqual(result['probabilities'][name][k],reference[k])
            self.assertAlmostEqual(result['ap' if name=='raw' else 'calibrated_ap'],reference['ap'])
            self.assertAlmostEqual(result['auroc' if name=='raw' else 'calibrated_auroc'],reference['auroc'])


if __name__=='__main__': unittest.main()
