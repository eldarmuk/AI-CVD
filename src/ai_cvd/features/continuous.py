"""Public tensor adapter for the retained Study A R0/R1 model definitions."""

from dataclasses import dataclass

import torch

PRIMITIVES = ("temperature", "heartrate", "sbp", "dbp", "saturation", "steps")
PROCESS_NAMES = (
    tuple("observed_" + f for f in PRIMITIVES)
    + tuple("recency_known_" + f for f in PRIMITIVES)
    + tuple("log_recency_" + f for f in PRIMITIVES)
    + (
        "hour_sin",
        "hour_cos",
        "is_night",
        "log_steps_interval",
        "steps_interval_known",
        "observed_steps_source",
        "steps_counter_reset",
    )
)


@dataclass
class View:
    values: torch.Tensor
    M: torch.Tensor
    A: torch.Tensor
    V: torch.Tensor
    process: torch.Tensor
    grid_minutes: int = 5


def make_view(values, observed, process, artificial=None):
    """Accept normalized tensors; this adapter is not the private source pipeline."""
    if values.ndim != 3 or values.shape[-1] != 6 or observed.shape != values.shape:
        raise ValueError("Expected B,T,6 values and matching mask")
    if observed.dtype != torch.bool or not torch.isfinite(values[observed]).all():
        raise ValueError("Boolean mask and finite observed values required")
    if process.shape != (*values.shape[:2], 25) or not torch.isfinite(process).all():
        raise ValueError("Expected finite B,T,25 process context")
    artificial = torch.zeros_like(observed) if artificial is None else artificial
    if (
        artificial.dtype != torch.bool
        or artificial.shape != observed.shape
        or (artificial & ~observed).any()
    ):
        raise ValueError("Artificial targets must be naturally observed")
    return View(values, observed, artificial, observed & ~artificial, process)
