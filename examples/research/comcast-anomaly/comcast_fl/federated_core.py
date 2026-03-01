"""Core federated data/model helpers shared by simulation and deployment wrappers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig, dumps_config_json
from .constants import NUM_BINS, NUM_CLASSES, V2_NODE_CONTEXT_FEATURES
from .model import SeqContextFusionModel, normalize_splits
from .synthetic import generate_client_domain_dataset
from .training import create_loaders

_CLIENT_CACHE: dict[str, dict] = {}
_SERVER_EVAL_CACHE: dict[str, dict] = {}


def _cache_key(cfg: ExperimentConfig, domain: str, client_id: str) -> str:
    return f"{domain}::{client_id}::{dumps_config_json(cfg)}"


def build_client_bundle(client_id: str, domain: str, cfg: ExperimentConfig) -> dict:
    """Build one client's local train/val/test bundle for one domain."""
    key = _cache_key(cfg, domain, client_id)
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    raw = generate_client_domain_dataset(client_id=client_id, signal_domain=domain, cfg=cfg)
    normed = normalize_splits(raw)
    loaders = create_loaders(normed, batch_size=cfg.local_training.batch_size)

    bundle = {
        "client_id": client_id,
        "domain": domain,
        "raw": raw,
        "normed": normed,
        "loaders": loaders,
        "context_dim": len(V2_NODE_CONTEXT_FEATURES),
    }
    _CLIENT_CACHE[key] = bundle
    return bundle


def build_server_eval_bundle(domain: str, cfg: ExperimentConfig) -> dict:
    """Build deterministic evaluation bundle for server-side validation and testing."""
    key = _cache_key(cfg, domain, "server_eval")
    if key in _SERVER_EVAL_CACHE:
        return _SERVER_EVAL_CACHE[key]

    # Use a fixed synthetic evaluator identity so metrics are deterministic.
    bundle = build_client_bundle(client_id="server_eval", domain=domain, cfg=cfg)
    _SERVER_EVAL_CACHE[key] = bundle
    return bundle


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(context_dim: int) -> SeqContextFusionModel:
    return SeqContextFusionModel(
        num_bins=NUM_BINS,
        context_dim=context_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=48,
    )


def get_client_identity(node_config: dict[str, Any], fallback: str) -> str:
    """Prefer explicit client-id; fallback to partition-id, then fallback string."""
    if "client-id" in node_config:
        return str(node_config["client-id"])
    if "partition-id" in node_config:
        return str(node_config["partition-id"])
    if "node-id" in node_config:
        return str(node_config["node-id"])
    return fallback


def format_run_result_for_summary(domain: str, metrics: dict, edge: dict) -> dict:
    return {
        "domain": domain,
        "unknown_threshold": float(metrics["unknown_threshold"]),
        "gated_macro_f1": float(metrics["gated_metrics"]["macro_f1"]),
        "gated_event_peak_macro_f1": float(metrics["gated_metrics"]["event_peak_macro_f1"]),
        "gated_impulse_f1": float(metrics["gated_metrics"]["impulse_f1"]),
        "gated_unknown_f1": float(metrics["gated_metrics"]["unknown_f1"]),
        "anomaly_auroc": float(metrics["gated_metrics"]["anomaly_auroc"]),
        "params": int(edge["params"]),
        "p95_latency_ms_cpu_proxy": float(edge["p95_latency_ms_cpu_proxy"]),
        "quantization_ready": bool(edge["quantization_ready"]),
        "pass_edge_gate": bool(edge["pass_edge_gate"]),
    }


def pretty_json(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
