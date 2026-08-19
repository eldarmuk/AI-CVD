"""Hybrid CGTA network fusing sequence trajectories and tabular summaries."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cgta_net import CGTANet


class HybridCGTANet(nn.Module):
    """Combine a circadian-gated sequence encoder with a tabular MLP branch."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        circadian_dim: int,
        tabular_dim: int,
        sequence_hidden_dim: int = 96,
        sequence_embedding_dim: int = 64,
        tabular_embedding_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if tabular_dim <= 0:
            raise ValueError("HybridCGTANet requires at least one tabular feature.")

        self.cgta = CGTANet(
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
            circadian_dim=circadian_dim,
            hidden_dim=sequence_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        encoded_dim = sequence_hidden_dim * 2
        self.sequence_projection = nn.Sequential(
            nn.Linear(encoded_dim + static_dim, sequence_embedding_dim),
            nn.LayerNorm(sequence_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, tabular_embedding_dim),
            nn.BatchNorm1d(tabular_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(sequence_embedding_dim + tabular_embedding_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(96, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def sequence_embedding(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        X_circadian: torch.Tensor,
    ) -> torch.Tensor:
        sequence_states, _ = self.cgta.encoder(X_dynamic)
        gate = self.cgta.time_gate(X_circadian)
        gated_states = sequence_states * gate
        attention_input = torch.cat([gated_states, X_circadian], dim=-1)
        attention_weights = torch.softmax(self.cgta.attention_score(attention_input), dim=1)
        temporal_context = torch.sum(gated_states * attention_weights, dim=1)
        return self.sequence_projection(torch.cat([temporal_context, X_static], dim=1))

    def forward(
        self,
        X_dynamic: torch.Tensor,
        X_static: torch.Tensor,
        X_circadian: torch.Tensor,
        X_tabular: torch.Tensor,
    ) -> torch.Tensor:
        seq_embedding = self.sequence_embedding(X_dynamic, X_static, X_circadian)
        tab_embedding = self.tabular_branch(X_tabular)
        fused = torch.cat([seq_embedding, tab_embedding], dim=1)
        return self.classifier(fused).squeeze(1)
