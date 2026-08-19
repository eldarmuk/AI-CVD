"""End-to-end supervised sequence classifier for imminent crisis prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.components.lstm_vae_model import LSTM_VAE


class ClinicalCrossAttentionBridge(nn.Module):
    """Use static clinical context as a query over encoded temporal states."""

    def __init__(self, hidden_dim: int, static_dim: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.static_query = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence_states: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        query = self.static_query(static_features).unsqueeze(1)
        attended, _ = self.attention(query, sequence_states, sequence_states)
        return self.norm(attended.squeeze(1) + query.squeeze(1))


class TimeAwareAttentionPool(nn.Module):
    """Learn a weighted temporal summary while preserving the final hidden state."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence_states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(sequence_states), dim=1)
        return torch.sum(sequence_states * weights, dim=1)


class SupervisedSequenceNet(nn.Module):
    """Decoder-free CCA-TAVAE-style classifier returning p(Crisis)."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        sequence_length: int = 96,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()
        if static_dim <= 0:
            raise ValueError("SupervisedSequenceNet requires at least one static feature.")

        vae_encoder = LSTM_VAE(
            input_dim=dynamic_dim,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        self.encoder_lstm = vae_encoder.encoder_lstm
        self.temporal_pool = TimeAwareAttentionPool(hidden_dim)
        self.cross_attention = ClinicalCrossAttentionBridge(hidden_dim, static_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3 + static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, X_dynamic: torch.Tensor, X_static: torch.Tensor) -> torch.Tensor:
        sequence_states, (hidden, _) = self.encoder_lstm(X_dynamic)
        final_hidden = hidden[-1]
        pooled_hidden = self.temporal_pool(sequence_states)
        clinical_context = self.cross_attention(sequence_states, X_static)
        features = torch.cat([final_hidden, pooled_hidden, clinical_context, X_static], dim=1)
        return self.classifier(features).squeeze(1)
