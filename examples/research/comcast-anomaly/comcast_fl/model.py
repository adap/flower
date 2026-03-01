"""Model architecture and data normalization helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SeqContextFusionModel(nn.Module):
    """Compact sequence encoder + context fusion model."""

    def __init__(self, num_bins: int, context_dim: int, num_classes: int, hidden_dim: int = 48):
        super().__init__()
        self.step_encoder = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)

        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        b, t, f = x_seq.shape
        z = x_seq.reshape(b * t, 1, f)
        z = self.step_encoder(z).squeeze(-1)
        z = z.reshape(b, t, -1)
        h, _ = self.gru(z)
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        seq_context = (h * weights).sum(dim=1)
        ctx = self.context_proj(x_ctx)
        return self.head(torch.cat([seq_context, ctx], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_splits(domain_data: dict) -> dict:
    """Normalize train/val/test splits using train-set statistics."""
    tr, va, te = domain_data["train"], domain_data["val"], domain_data["test"]

    static_mean = tr["X_static"].mean(axis=0, keepdims=True)
    static_std = tr["X_static"].std(axis=0, keepdims=True) + 1e-6

    seq_mean = tr["X_seq"].mean(axis=(0, 1), keepdims=True)
    seq_std = tr["X_seq"].std(axis=(0, 1), keepdims=True) + 1e-6

    ctx_mean = tr["X_context"].mean(axis=0, keepdims=True)
    ctx_std = tr["X_context"].std(axis=0, keepdims=True) + 1e-6

    def norm(split: dict) -> dict:
        return {
            "X_static_n": ((split["X_static"] - static_mean) / static_std).astype(np.float32),
            "X_seq_n": ((split["X_seq"] - seq_mean) / seq_std).astype(np.float32),
            "X_context_n": ((split["X_context"] - ctx_mean) / ctx_std).astype(np.float32),
            "y": split["y"].astype(np.int64),
            "regime_ids": split["regime_ids"].astype(np.int64),
        }

    return {
        "train": norm(tr),
        "val": norm(va),
        "test": norm(te),
        "stats": {
            "static_mean": static_mean.astype(np.float32),
            "static_std": static_std.astype(np.float32),
            "seq_mean": seq_mean.astype(np.float32),
            "seq_std": seq_std.astype(np.float32),
            "ctx_mean": ctx_mean.astype(np.float32),
            "ctx_std": ctx_std.astype(np.float32),
        },
    }
