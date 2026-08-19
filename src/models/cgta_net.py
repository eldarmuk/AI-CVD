"""Circadian-Gated Temporal Attention Network for clinical deterioration."""

from __future__ import annotations

import torch
import torch.nn as nn


class CGTANet(nn.Module):
    """
    Fuse physiological trajectories with an explicit circadian modulation gate.

    The recurrent encoder models the 96-step physiological sequence. Circadian
    features at each step produce a sigmoid gate that rescales the encoded
    hidden states before temporal attention pooling and static clinical fusion.
    """

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        circadian_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if dynamic_dim <= 0:
            raise ValueError("CGTANet requires at least one dynamic feature.")
        if static_dim <= 0:
            raise ValueError("CGTANet requires at least one static feature.")
        if circadian_dim <= 0:
            raise ValueError("CGTANet requires circadian features for gating.")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=dynamic_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout,
        )
        encoded_dim = hidden_dim * 2
        self.time_gate = nn.Sequential(
            nn.Linear(circadian_dim, encoded_dim),
            nn.Sigmoid(),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(encoded_dim + circadian_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim + static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        X_circadian: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier_embedding(X_dynamic, X_static, X_circadian, include_score=True)

    def temporal_context(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        X_circadian: torch.Tensor,
    ) -> torch.Tensor:
        sequence_states, _ = self.encoder(X_dynamic)
        gate = self.time_gate(X_circadian)
        gated_states = sequence_states * gate
        attention_input = torch.cat([gated_states, X_circadian], dim=-1)
        attention_weights = torch.softmax(self.attention_score(attention_input), dim=1)
        temporal_context = torch.sum(gated_states * attention_weights, dim=1)
        return torch.cat([temporal_context, X_static], dim=1)

    def classifier_embedding(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        X_circadian: torch.Tensor,
        include_score: bool = False,
    ) -> torch.Tensor:
        features = self.temporal_context(X_dynamic, X_static, X_circadian)
        embedding = self.classifier[:-2](features)
        if include_score:
            return self.classifier[-2:](embedding).squeeze(1)
        return embedding

    def gate_values(self, X_circadian: torch.Tensor) -> torch.Tensor:
        """Return per-timestep gate activations for explainability."""
        return self.time_gate(X_circadian)
