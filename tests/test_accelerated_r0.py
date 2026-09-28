import unittest
import numpy as np
import torch
from scripts.accelerated_r0 import score_patient,check_equivalence
from src.final_study.config import load
from src.architecture_study.synthetic import tensors
from src.ai_cvd.features import FEATURE_NAMES


class ToyRisk(torch.nn.Module):
    def __init__(self,large_error=False):
        super().__init__(); self.anchor=torch.nn.Parameter(torch.zeros(1)); self.large_error=large_error
    def forward(self,view):
        n=len(view.values)
        # Deliberately introduce a deterministic batch-size-dependent numeric drift.
        shift=(.1 if self.large_error else 1e-9) if n>64 else 0.
        return torch.full((n,),shift,dtype=torch.float64)


class AccelerationTests(unittest.TestCase):
    def test_threshold_nearby_batches_replayed_at_reference_size(self):
        _,_,task=load(); x=tensors(6).numpy().reshape(-1,len(FEATURE_NAMES))
        ends=np.arange(95,400)
        p,g=score_patient(ToyRisk(),x,ends,{'mean':[0]*6,'scale':[1]*6},task,.5,0.)
        np.testing.assert_array_equal(p,np.full(len(ends),.5))
        self.assertEqual(g['guarded_reference_batches'],5)
        self.assertGreater(g['near_threshold_windows'],0)

    def test_canary_tolerance_and_order_fail_closed(self):
        _,_,task=load(); x=tensors(6).numpy().reshape(-1,len(FEATURE_NAMES))
        with self.assertRaises(ValueError):
            score_patient(ToyRisk(True),x,np.arange(95,400),{'mean':[0]*6,'scale':[1]*6},task,.9,0.)
        with self.assertRaises(ValueError):
            score_patient(ToyRisk(),x,[100,99],{'mean':[0]*6,'scale':[1]*6},task,.9,0.)
        with self.assertRaises(ValueError): check_equivalence(np.array([np.nan]),np.array([.1]))


if __name__=='__main__': unittest.main()
