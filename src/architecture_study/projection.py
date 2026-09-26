"""Explicit primitive projection; derived magnitudes are never encoder inputs."""
from dataclasses import dataclass
import numpy as np
import torch
from src.ai_cvd.features import FEATURE_NAMES

PRIMITIVES = ('temperature', 'heartrate', 'sbp', 'dbp', 'saturation', 'steps')
VALUE_INDICES = tuple(FEATURE_NAMES.index(f) for f in PRIMITIVES)
PROCESS_NAMES = (tuple('observed_' + f for f in PRIMITIVES) +
                 tuple('recency_known_' + f for f in PRIMITIVES) +
                 tuple('log_recency_' + f for f in PRIMITIVES) +
                 ('hour_sin', 'hour_cos', 'is_night', 'log_steps_interval',
                  'steps_interval_known', 'observed_steps_source', 'steps_counter_reset'))


@dataclass
class View:
    values: torch.Tensor
    M: torch.Tensor
    A: torch.Tensor
    V: torch.Tensor
    process: torch.Tensor
    grid_minutes: int


def mask_observed(M, generator, fraction=0.2):
    """One visible and one target when possible; empty/single-reading rows are safe."""
    A = torch.zeros_like(M, dtype=torch.bool)
    for i in range(len(M)):
        indices = M[i].flatten().nonzero().flatten()
        if len(indices) < 2:
            continue
        n = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        chosen = indices[torch.randperm(len(indices), generator=generator, device=M.device)[:n]]
        A[i].flatten()[chosen] = True
    return A


def project(X, mean, scale, task, A=None, feature_names=FEATURE_NAMES):
    if tuple(feature_names) != FEATURE_NAMES or X.ndim != 3 or X.shape[1:] != (task.sequence_steps, len(FEATURE_NAMES)):
        raise ValueError('Canonical feature order or shape mismatch')
    values = X[..., list(VALUE_INDICES)]
    M = torch.stack([X[..., FEATURE_NAMES.index('observed_' + f)] for f in PRIMITIVES], -1)
    if not torch.all((M == 0) | (M == 1)) or not torch.equal(M.bool(), torch.isfinite(values)):
        raise ValueError('Primitive validity/mask mismatch')
    M = M.bool()
    A = torch.zeros_like(M) if A is None else A
    if A.dtype != torch.bool or A.shape != M.shape or torch.any(A & ~M):
        raise ValueError('Artificial targets must be naturally observed')
    mean = torch.as_tensor(mean, dtype=X.dtype, device=X.device)
    scale = torch.as_tensor(scale, dtype=X.dtype, device=X.device)
    if mean.shape != (6,) or scale.shape != (6,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid primitive scaler')
    recency = torch.stack([X[..., FEATURE_NAMES.index('time_since_last_' + f)] for f in PRIMITIVES], -1)
    known = torch.isfinite(recency)
    if (recency[known] < 0).any() or torch.isinf(recency).any():
        raise ValueError('Invalid recency')
    clock = X[..., [FEATURE_NAMES.index(f) for f in ('hour_sin', 'hour_cos', 'is_night')]]
    interval = X[..., FEATURE_NAMES.index('steps_delta_interval_minutes')].unsqueeze(-1)
    iknown = torch.isfinite(interval)
    if (interval[iknown] < 0).any() or torch.isinf(interval).any():
        raise ValueError('Invalid step interval')
    flags = X[..., [FEATURE_NAMES.index(f) for f in ('observed_steps_source', 'steps_counter_reset')]]
    process = torch.cat((M.to(X.dtype), known.to(X.dtype),
                         torch.log1p(torch.where(known, recency, 0)), clock,
                         torch.log1p(torch.where(iknown, interval, 0)), iknown.to(X.dtype), flags), -1)
    if not torch.isfinite(process).all() or not torch.all((flags == 0) | (flags == 1)):
        raise ValueError('Invalid process context')
    return View((values - mean) / scale, M, A, M & ~A, process, task.grid_minutes)


def prefix_window(rows, bucket_start_ns, prediction_ns, task):
    """Reject wrong/duplicate grid rows; future rows cannot influence the projection."""
    starts = np.asarray(bucket_start_ns, dtype=np.int64)
    step = task.grid_minutes * 60 * 10**9
    select = (starts >= prediction_ns - task.lookback_minutes * 60 * 10**9) & (starts + step <= prediction_ns)
    expected = prediction_ns - task.lookback_minutes * 60 * 10**9 + np.arange(task.sequence_steps) * step
    if not np.array_equal(starts[select], expected):
        raise ValueError('Incomplete, unordered or duplicate prediction prefix')
    return np.asarray(rows)[select].copy()
