from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import iqr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader


def rc_get(rc: Dict[str, Any], *keys: str, default=None):
    """Return first existing key from rc, else default."""
    for k in keys:
        if k in rc:
            return rc[k]
    return default


@dataclass
class TrainConfig:
    batch_size: int
    local_epochs: int
    learning_rate: float
    early_stop_patience: int


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target RMSE (no sklearn dependency)."""
    if y_true.size == 0:
        return np.zeros((0,), dtype=float)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    return np.sqrt(mse).astype(float)


def rpiq(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r = rmse(y_true, y_pred)
    out = []
    for j in range(y_true.shape[1]):
        denom = r[j] if r[j] != 0 else np.nan
        out.append(iqr(y_true[:, j]) / denom)
    return np.array(out, dtype=float)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = []
    for j in range(y_true.shape[1]):
        out.append(r2_score(y_true[:, j], y_pred[:, j]))
    return np.array(out, dtype=float)


def to_weights(model: torch.nn.Module):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def from_weights(model: torch.nn.Module, weights):
    sd = model.state_dict()
    new_sd = {}
    for (k, _), w in zip(sd.items(), weights):
        new_sd[k] = torch.tensor(w)
    model.load_state_dict(new_sd, strict=True)


def train_local(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> float:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    best_state = None
    patience = 0

    for _epoch in range(cfg.local_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vlosses.append(criterion(pred, yb).item())

        avg_val = float(np.mean(vlosses)) if vlosses else float("inf")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        if patience >= cfg.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return float(best_val)


def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 1))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
