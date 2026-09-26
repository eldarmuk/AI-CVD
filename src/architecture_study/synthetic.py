"""Fictional tensors only; no private data are embedded in test fixtures."""
import numpy as np
import torch
from src.ai_cvd.features import FEATURE_NAMES
from .projection import PRIMITIVES


def tensors(batch=8, seed=13, observed_fraction=0.15):
    rng = np.random.default_rng(seed)
    x = np.full((batch, 96, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for name in FEATURE_NAMES:
        if name.startswith('observed_') or name.startswith('known_') or name in ('steps_counter_reset', 'is_night'):
            x[..., FEATURE_NAMES.index(name)] = 0
    clock = np.arange(96) / 288 * 2 * np.pi
    x[..., FEATURE_NAMES.index('hour_sin')] = np.sin(clock)
    x[..., FEATURE_NAMES.index('hour_cos')] = np.cos(clock)
    for name in PRIMITIVES:
        m = rng.random((batch, 96)) < observed_fraction
        values = rng.normal(size=(batch, 96))
        if name == 'steps':
            values = rng.integers(0, 20, (batch, 96)).astype(float)
        x[..., FEATURE_NAMES.index(name)] = np.where(m, values, np.nan)
        x[..., FEATURE_NAMES.index('observed_' + name)] = m
        for b in range(batch):
            last = None
            for j in range(96):
                if m[b, j]:
                    last = j
                if last is not None:
                    x[b, j, FEATURE_NAMES.index('time_since_last_' + name)] = (j-last)*5
    return torch.from_numpy(x)
