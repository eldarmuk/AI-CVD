"""Learned input/hidden decay GRU-D. See B2 implementation equations."""

import torch
from torch import nn


class GRUD(nn.Module):
    def __init__(self, channels, context=56, hidden=32, head=16, dropout=0.1):
        super().__init__()
        if channels not in (7, 13) or (context, hidden, head, dropout) != (56, 32, 16, 0.1):
            raise ValueError("Only the two frozen GRU-D architectures are allowed")
        self.channels, self.hidden = channels, hidden
        # Diagonal input decay and full channel-to-hidden decay.
        self.input_weight = nn.Parameter(torch.ones(channels))
        self.input_bias = nn.Parameter(torch.zeros(channels))
        self.hidden_decay = nn.Linear(channels, hidden)
        self.gates = nn.Linear(2 * channels + hidden, 2 * hidden)
        self.candidate = nn.Linear(2 * channels + hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + context, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1)
        )

    def step(self, value, mask, delta, target, last, hidden):
        gx = torch.exp(-torch.relu(self.input_weight * delta + self.input_bias))
        gh = torch.exp(-torch.relu(self.hidden_decay(delta)))
        decayed = gh * hidden
        observed = mask.bool()
        # torch.where prevents NaN/hidden placeholders entering either cache or input.
        x = torch.where(observed, value, gx * last + (1 - gx) * target)
        z, r = torch.sigmoid(self.gates(torch.cat((x, decayed, mask), -1))).chunk(2, -1)
        candidate = torch.tanh(self.candidate(torch.cat((x, r * decayed, mask), -1)))
        h = (1 - z) * decayed + z * candidate
        return h, torch.where(observed, value, last), gx, gh

    def forward(self, values, masks, deltas, admitted, target, context):
        if (
            values.shape != masks.shape
            or values.shape != deltas.shape
            or values.shape[-1] != self.channels
        ):
            raise ValueError("Temporal input dimensions differ")
        if torch.any(deltas < 0) or not torch.isfinite(deltas).all():
            raise ValueError("Nonnegative finite elapsed hours required")
        if torch.any((masks != 0) & (masks != 1)) or torch.any(
            masks.bool() & ~admitted.bool().unsqueeze(-1)
        ):
            raise ValueError("Invalid observation/padding mask")
        h = values.new_zeros((len(values), self.hidden))
        last = target.clone()
        for t in range(values.shape[1]):
            new_h, new_last, _, _ = self.step(
                values[:, t], masks[:, t], deltas[:, t], target, last, h
            )
            active = admitted[:, t].bool().unsqueeze(-1)
            h, last = torch.where(active, new_h, h), torch.where(active, new_last, last)
        return self.head(torch.cat((h, context), -1)).squeeze(-1)


def make_model(family, config):
    m = config["models"]
    if family not in ("grud", "relative_grud"):
        raise ValueError("Unapproved temporal family")
    model = GRUD(
        7 if family == "grud" else 13,
        config["feature_subsets"][m["neural_context_subset"]]["encoded_dimension"],
        m["grud_hidden"],
        m["grud_head_hidden"],
        m["neural_dropout"],
    )
    if sum(p.numel() for p in model.parameters()) > m["neural_max_parameters"]:
        raise ValueError("Parameter ceiling exceeded")
    return model
