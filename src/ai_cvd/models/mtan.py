"""Frozen small mTAN adaptation, not the published benchmark architecture."""

import math

import torch
from torch import nn


def masked_attention(query, key, values, observed):
    """B,H,Q,D and B,H,T,D; variable-specific normalization over T only."""
    logits = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    valid = observed[:, None, None, :, :].bool()
    logits = logits.unsqueeze(-1).expand(-1, -1, -1, -1, values.shape[-1])
    nonempty = valid.any(dim=-2, keepdim=True)
    # Empty channels use a finite softmax input, then exact zero weights.
    logits = logits.masked_fill(~valid, -torch.inf)
    logits = torch.where(nonempty, logits, torch.zeros_like(logits))
    weights = torch.softmax(logits, dim=-2) * valid
    clean = torch.where(observed.bool(), values, torch.zeros_like(values))
    interpolated = (weights * clean[:, None, None, :, :]).sum(-2).mean(1)
    support = observed.bool().any(1)[:, None, :].expand(-1, query.shape[2], -1)
    return interpolated, support.to(values.dtype), weights


class MTAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_time = nn.Linear(1, 1)
        self.periodic_time = nn.Linear(1, 15)
        self.query = nn.Linear(16, 16)
        self.key = nn.Linear(16, 16)
        self.gru = nn.GRU(14, 32, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(32 + 56, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1)
        )

    def embed(self, times):
        t = times.unsqueeze(-1)
        return torch.cat((self.linear_time(t), torch.sin(self.periodic_time(t))), -1)

    def forward(self, values, masks, admitted, context, key_times, reference_times):
        if (
            values.shape != masks.shape
            or values.shape[1:] != (288, 7)
            or context.shape[1:] != (56,)
        ):
            raise ValueError("Frozen mTAN dimensions differ")
        if key_times.shape != admitted.shape or reference_times.shape != (len(values), 48):
            raise ValueError("Time grid dimensions differ")
        if torch.any((masks != 0) & (masks != 1)) or torch.any(
            masks.bool() & ~admitted.bool().unsqueeze(-1)
        ):
            raise ValueError("Invalid observation/padding mask")
        if (
            not torch.isfinite(key_times).all()
            or not torch.isfinite(reference_times).all()
            or torch.any(key_times[admitted] >= 0)
            or torch.any(reference_times >= 0)
        ):
            raise ValueError("Finite strictly pre-alarm times required")
        if not torch.isfinite(values[masks.bool()]).all():
            raise ValueError("Observed values must be finite")
        # Padding has no attention key, including its time embedding.
        times = torch.where(admitted, key_times, torch.zeros_like(key_times))
        q = self.query(self.embed(reference_times)).reshape(len(values), 48, 4, 4).transpose(1, 2)
        k = self.key(self.embed(times)).reshape(len(values), 288, 4, 4).transpose(1, 2)
        v, support, _ = masked_attention(q, k, values, masks.bool() & admitted.unsqueeze(-1))
        _, hidden = self.gru(torch.cat((v, support), -1))
        return self.head(torch.cat((hidden[0], context), -1)).squeeze(-1)


def make_model(family, config):
    m = config["models"]
    if family != "mtan" or (
        m["mtan_time_embedding_dim"],
        m["mtan_attention_heads"],
        m["mtan_query_key_dim"],
        m["mtan_reference_times"],
    ) != (16, 4, 16, 48):
        raise ValueError("Frozen mTAN settings differ")
    model = MTAN()
    if sum(p.numel() for p in model.parameters()) > m["neural_max_parameters"]:
        raise ValueError("Parameter ceiling exceeded")
    return model
