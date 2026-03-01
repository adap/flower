"""Training and evaluation helpers."""

from __future__ import annotations

import copy
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import V2_CLASS_NAMES, V2_EDGE_TARGETS, V2_REGIMES


def create_loaders(normed: dict, batch_size: int) -> dict[str, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(normed["train"]["X_seq_n"]),
        torch.from_numpy(normed["train"]["X_context_n"]),
        torch.from_numpy(normed["train"]["y"]),
        torch.from_numpy(normed["train"]["regime_ids"]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(normed["val"]["X_seq_n"]),
        torch.from_numpy(normed["val"]["X_context_n"]),
        torch.from_numpy(normed["val"]["y"]),
        torch.from_numpy(normed["val"]["regime_ids"]),
    )
    test_ds = TensorDataset(
        torch.from_numpy(normed["test"]["X_seq_n"]),
        torch.from_numpy(normed["test"]["X_context_n"]),
        torch.from_numpy(normed["test"]["y"]),
        torch.from_numpy(normed["test"]["regime_ids"]),
    )
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    }


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_true, all_regime = [], [], []
    with torch.no_grad():
        for x_seq, x_ctx, yb, rb in loader:
            logits = model(x_seq.to(device), x_ctx.to(device))
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_true.append(yb.numpy())
            all_regime.append(rb.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true), np.concatenate(all_regime)


def train_local_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    local_epochs: int,
) -> tuple[float, int]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    running_loss = 0.0
    n_seen = 0

    for _ in range(local_epochs):
        for x_seq, x_ctx, yb, _ in loader:
            x_seq = x_seq.to(device)
            x_ctx = x_ctx.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_seq, x_ctx)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = x_seq.shape[0]
            running_loss += float(loss.item()) * bs
            n_seen += int(bs)

    return running_loss / max(1, n_seen), n_seen


def choose_unknown_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    unknown_idx: int,
    grid_size: int,
) -> float:
    score = 1.0 - probs.max(axis=1)
    target = (y_true == unknown_idx).astype(np.int64)
    if target.min() == target.max():
        return 1.0

    grid = np.linspace(0.0, 1.0, grid_size)
    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        pred_u = (score >= t).astype(np.int64)
        f1 = f1_score(target, pred_u, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def apply_unknown_gate(probs: np.ndarray, raw_pred: np.ndarray, threshold: float, unknown_idx: int) -> np.ndarray:
    unknown_score = 1.0 - probs.max(axis=1)
    gated = raw_pred.copy()
    gated[unknown_score >= threshold] = unknown_idx
    return gated


def metrics_from_preds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    regime_ids: np.ndarray,
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=V2_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(V2_CLASS_NAMES)))

    anomaly_target = (y_true != 0).astype(np.int64)
    anomaly_score = 1.0 - probs.max(axis=1)
    if anomaly_target.min() == anomaly_target.max():
        anomaly_auroc = float("nan")
    else:
        anomaly_auroc = float(roc_auc_score(anomaly_target, anomaly_score))

    by_regime = {}
    for r, name in enumerate(V2_REGIMES):
        idx = np.where(regime_ids == r)[0]
        by_regime[name] = float("nan") if idx.size == 0 else float(f1_score(y_true[idx], y_pred[idx], average="macro"))

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float((y_true == y_pred).mean()),
        "anomaly_auroc": anomaly_auroc,
        "event_peak_macro_f1": by_regime.get("event_peak", float("nan")),
        "by_regime_macro_f1": by_regime,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "impulse_f1": float(report.get("impulse_noise", {}).get("f1-score", 0.0)),
        "unknown_f1": float(report.get("unknown_other", {}).get("f1-score", 0.0)),
    }


def measure_latency_p95_cpu_ms(model: nn.Module, seq_sample: np.ndarray, ctx_sample: np.ndarray) -> float:
    cpu_model = copy.deepcopy(model).cpu().eval()
    seq_t = torch.from_numpy(seq_sample[None, ...].astype(np.float32))
    ctx_t = torch.from_numpy(ctx_sample[None, ...].astype(np.float32))

    with torch.no_grad():
        for _ in range(50):
            _ = cpu_model(seq_t, ctx_t)
        times = []
        for _ in range(500):
            t0 = time.perf_counter()
            _ = cpu_model(seq_t, ctx_t)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.percentile(np.array(times, dtype=np.float32), 95.0))


def quantization_readiness(model: nn.Module, seq_sample: np.ndarray, ctx_sample: np.ndarray) -> tuple[bool, str | None]:
    seq_t = torch.from_numpy(seq_sample[None, ...].astype(np.float32))
    ctx_t = torch.from_numpy(ctx_sample[None, ...].astype(np.float32))

    quantize_dynamic = getattr(torch.quantization, "quantize_dynamic", None)
    if quantize_dynamic is None:
        try:
            from torch.ao.quantization import quantize_dynamic as qdyn

            quantize_dynamic = qdyn
        except Exception:
            return False, "quantize_dynamic_not_available"

    try:
        cpu_model = copy.deepcopy(model).cpu().eval()
        qmodel = quantize_dynamic(cpu_model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        with torch.no_grad():
            _ = qmodel(seq_t, ctx_t)
        return True, None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, str(exc)


def evaluate_with_gate(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    unknown_idx: int,
    grid_size: int,
) -> dict:
    val_probs, val_true, _ = collect_probs(model, val_loader, device)
    threshold = choose_unknown_threshold(val_probs, val_true, unknown_idx=unknown_idx, grid_size=grid_size)

    probs, y_true, regime_ids = collect_probs(model, test_loader, device)
    raw_pred = np.argmax(probs, axis=1)
    gated_pred = apply_unknown_gate(probs, raw_pred, threshold, unknown_idx=unknown_idx)

    raw_metrics = metrics_from_preds(y_true, raw_pred, probs, regime_ids)
    gated_metrics = metrics_from_preds(y_true, gated_pred, probs, regime_ids)

    return {
        "unknown_threshold": float(threshold),
        "raw_metrics": raw_metrics,
        "gated_metrics": gated_metrics,
    }


def evaluate_edge_constraints(model: nn.Module, seq_sample: np.ndarray, ctx_sample: np.ndarray, params: int) -> dict:
    p95 = measure_latency_p95_cpu_ms(model, seq_sample, ctx_sample)
    qready, qerr = quantization_readiness(model, seq_sample, ctx_sample)

    edge_pass = (
        params <= V2_EDGE_TARGETS["max_params"]
        and p95 <= V2_EDGE_TARGETS["p95_latency_ms_cpu_proxy"]
        and bool(qready) == bool(V2_EDGE_TARGETS["quantization_ready"])
    )

    return {
        "params": int(params),
        "p95_latency_ms_cpu_proxy": float(p95),
        "quantization_ready": bool(qready),
        "quantization_error": qerr,
        "pass_edge_gate": bool(edge_pass),
    }
