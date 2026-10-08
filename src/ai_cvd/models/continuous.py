"""Deterministic R0/R1 only. Missing values never become cache observations."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from ai_cvd.features.continuous import PROCESS_NAMES


class EvidencePool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, states, valid):
        logits = self.score(states).squeeze(-1).masked_fill(~valid, -1e9)
        weights = logits.softmax(-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return (states * weights.unsqueeze(-1)).sum(1)


class ValueEncoder(nn.Module):
    """Shared GRU-D-style core; not claimed as a verbatim GRU-D reproduction.

    Visible age uses closed buckets. Natural recency remains separate context.
    A wholly invisible row decays state without a GRU update from placeholders.
    """

    def __init__(self, width):
        super().__init__()
        self.input_decay = nn.Parameter(torch.full((6,), -4.0))
        self.hidden_decay = nn.Parameter(torch.full((width, 6), -6.0))
        self.cell = nn.GRUCell(42, width)  # values, V, visible age/known, natural M/age/known
        self.pool = EvidencePool(width)
        self.width = width

    def forward(self, view, trace=False):
        B, T, _ = view.values.shape
        h = view.values.new_zeros(B, self.width)
        cache = view.values.new_zeros(B, 6)
        last = torch.full_like(cache, -1)
        states, caches, ages, knowns = [], [], [], []
        for j in range(T):
            visible = view.V[:, j]
            known = visible | (last >= 0)
            age = torch.where(visible, 0, torch.where(last >= 0, (j - last) * view.grid_minutes, 0))
            gap = torch.log1p(age)
            gamma = torch.exp(-F.softplus(self.input_decay) * gap)
            # Only visible values can affect values/cache; hidden target magnitudes are inaccessible.
            value = torch.where(visible, view.values[:, j], gamma * cache)
            decayed = h * torch.exp(-F.linear(gap, F.softplus(self.hidden_decay)))
            inputs = torch.cat(
                (
                    value,
                    visible.to(value.dtype),
                    gap,
                    known.to(value.dtype),
                    view.process[:, j, :18],
                ),
                -1,
            )
            proposed = self.cell(inputs, decayed)
            h = torch.where(visible.any(-1, keepdim=True), proposed, decayed)
            cache = torch.where(visible, view.values[:, j], cache)
            last = torch.where(visible, j, last)
            states.append(h)
            if trace:
                caches.append(cache)
                ages.append(age)
                knowns.append(known)
        states = torch.stack(states, 1)
        pooled = self.pool(states, view.V.any(-1))
        if trace:
            return pooled, {
                "cache": torch.stack(caches, 1),
                "age": torch.stack(ages, 1),
                "known": torch.stack(knowns, 1),
                "states": states,
            }
        return pooled


class Encoder(nn.Module):
    def __init__(self, recipe, value_width=32, process_width=16):
        super().__init__()
        if recipe not in ("R0", "R1"):
            raise ValueError("Only R0 and R1 are implemented")
        self.recipe = recipe
        self.value = ValueEncoder(value_width)
        if recipe == "R1":
            self.process = nn.GRU(len(PROCESS_NAMES), process_width, batch_first=True)
            self.process_pool = EvidencePool(process_width)
        # R0 retains a fixed process summary for transparent all-empty fallback,
        # but has only one learned temporal encoder; both recipes access the same data.
        self.output_dim = (
            value_width + (process_width if recipe == "R1" else len(PROCESS_NAMES)) + 2
        )

    def forward(self, view):
        value = self.value(view)
        if self.recipe == "R1":
            states, _ = self.process(view.process)
            process = self.process_pool(
                states, torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
            )
        else:
            process = view.process.mean(1)
        evidence = torch.stack((~view.V.flatten(1).any(1), view.V.float().mean((1, 2))), -1).to(
            value.dtype
        )
        return torch.cat((value, process, evidence), -1)


class MaskedModel(nn.Module):
    def __init__(self, recipe, value_width=32, process_width=16, decoder_width=32):
        super().__init__()
        self.encoder = Encoder(recipe, value_width, process_width)
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 1, decoder_width),
            nn.Tanh(),
            nn.Linear(decoder_width, 6),
        )

    def forward(self, view):
        z = self.encoder(view)
        times = torch.linspace(0, 1, view.values.shape[1], device=z.device, dtype=z.dtype)
        expanded = z[:, None].expand(-1, len(times), -1)
        return self.decoder(torch.cat((expanded, times[None, :, None].expand(len(z), -1, -1)), -1))


def masked_huber(reconstruction, view, delta=1.0):
    """Mean of within-window/channel target means, including safe empty batches."""
    if reconstruction.shape != view.values.shape or not torch.isfinite(reconstruction).all():
        raise ValueError("Invalid reconstruction")
    target = view.A & view.M
    # Indexed selection ensures a naturally missing NaN cannot enter arithmetic.
    errors = torch.zeros_like(reconstruction)
    errors[target] = F.huber_loss(
        reconstruction[target], view.values[target], reduction="none", delta=delta
    )
    n = target.sum(1)
    contributing = n > 0
    per_channel = errors.sum(1) / n.clamp_min(1)
    return per_channel[contributing].mean() if contributing.any() else reconstruction.sum() * 0


class FrozenRisk(nn.Module):
    """A copied, frozen encoder plus logistic head; decoder genuinely absent."""

    def __init__(self, pretrained):
        super().__init__()
        self.encoder = copy.deepcopy(pretrained.encoder).requires_grad_(False).eval()
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def embeddings(self, view):
        if view.A.any():
            raise ValueError("Risk inference must use the uncorrupted clinical prefix")
        with torch.no_grad():
            return self.encoder(view)

    def forward(self, view):
        return self.head(self.embeddings(view)).squeeze(-1)


def parameter_counts(model):
    risk = FrozenRisk(model)
    return {
        "ssl_total": sum(p.numel() for p in model.parameters()),
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "risk_total": sum(p.numel() for p in risk.parameters()),
        "risk_trainable": sum(p.numel() for p in risk.parameters() if p.requires_grad),
    }
