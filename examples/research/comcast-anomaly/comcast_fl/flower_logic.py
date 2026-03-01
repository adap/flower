"""Shared Flower client/server handlers for Comcast anomaly FL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from torch import nn

from .config import ExperimentConfig
from .constants import V2_ACCEPTANCE_TARGETS, V2_EDGE_TARGETS
from .federated_core import (
    build_client_bundle,
    build_server_eval_bundle,
    format_run_result_for_summary,
    get_client_identity,
    get_device,
    make_model,
)
from .model import count_parameters
from .training import (
    collect_probs,
    evaluate_edge_constraints,
    evaluate_with_gate,
    metrics_from_preds,
    train_local_epoch,
)
from .utils import ensure_dir, to_builtin


def client_train_handler(msg: Message, context: Context, cfg: ExperimentConfig, domain: str) -> Message:
    client_id = get_client_identity(context.node_config, fallback=str(context.node_id))
    bundle = build_client_bundle(client_id=client_id, domain=domain, cfg=cfg)

    model = make_model(context_dim=bundle["context_dim"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    fit_cfg = msg.content.get("config", ConfigRecord())
    local_epochs = int(fit_cfg.get("local-epochs", cfg.local_training.local_epochs))
    lr = float(fit_cfg.get("lr", cfg.local_training.lr))
    wd = float(fit_cfg.get("weight-decay", cfg.local_training.weight_decay))

    device = get_device()
    model = model.to(device)

    train_loss, n_seen = train_local_epoch(
        model=model,
        loader=bundle["loaders"]["train"],
        device=device,
        lr=lr,
        weight_decay=wd,
        local_epochs=local_epochs,
    )

    probs, y_true, regime_ids = collect_probs(model, bundle["loaders"]["val"], device)
    val_macro_f1 = float(metrics_from_preds(y_true, np.argmax(probs, axis=1), probs, regime_ids)["macro_f1"])

    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord(
        {
            "train_loss": float(train_loss),
            "val_macro_f1": val_macro_f1,
            "num-examples": int(n_seen),
        }
    )
    content = RecordDict({"arrays": arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)


def client_evaluate_handler(msg: Message, context: Context, cfg: ExperimentConfig, domain: str) -> Message:
    client_id = get_client_identity(context.node_config, fallback=str(context.node_id))
    bundle = build_client_bundle(client_id=client_id, domain=domain, cfg=cfg)

    model = make_model(context_dim=bundle["context_dim"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    device = get_device()
    model = model.to(device)

    loader = bundle["loaders"]["test"]
    probs, y_true, regime_ids = collect_probs(model, loader, device)
    pred = np.argmax(probs, axis=1)
    eval_metrics = metrics_from_preds(y_true, pred, probs, regime_ids)

    metrics = MetricRecord(
        {
            "eval_macro_f1": float(eval_metrics["macro_f1"]),
            "eval_accuracy": float(eval_metrics["accuracy"]),
            "num-examples": int(len(loader.dataset)),
        }
    )
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)


def _domain_output_paths(cfg: ExperimentConfig, domain: str) -> dict[str, Path]:
    root = Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / domain
    ensure_dir(root)
    return {
        "root": root,
        "metrics": root / "metrics.json",
        "checkpoint": root / "checkpoint_best.pt",
        "cm": root / "confusion_matrix.npy",
        "threshold": root / "threshold.json",
        "contract": root / "contract.json",
    }


def server_main(grid: Grid, context: Context, cfg: ExperimentConfig, domain: str) -> dict[str, Any]:
    """Run one domain FL training and save domain artifacts."""
    fcfg = cfg.federation

    strategy = FedAvg(
        fraction_train=float(fcfg.fraction_train),
        fraction_evaluate=float(fcfg.fraction_evaluate),
        min_train_nodes=int(fcfg.min_train_nodes),
        min_evaluate_nodes=int(fcfg.min_evaluate_nodes),
        min_available_nodes=int(max(fcfg.min_train_nodes, fcfg.min_evaluate_nodes)),
    )

    context_dim = 12
    model = make_model(context_dim=context_dim)
    arrays = ArrayRecord(model.state_dict())

    # One deterministic server-side evaluator for round-wise visibility.
    server_eval = build_server_eval_bundle(domain=domain, cfg=cfg)
    device = get_device()

    def _evaluate_fn(_: int, arr: ArrayRecord) -> MetricRecord:
        m = make_model(context_dim=context_dim)
        m.load_state_dict(arr.to_torch_state_dict(), strict=True)
        m = m.to(device)
        probs, y_true, regime_ids = collect_probs(m, server_eval["loaders"]["val"], device)
        pred = np.argmax(probs, axis=1)
        metrics = metrics_from_preds(y_true, pred, probs, regime_ids)
        return MetricRecord({"server_val_macro_f1": float(metrics["macro_f1"])})

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=int(fcfg.num_rounds),
        train_config=ConfigRecord(
            {
                "local-epochs": int(cfg.local_training.local_epochs),
                "lr": float(cfg.local_training.lr),
                "weight-decay": float(cfg.local_training.weight_decay),
            }
        ),
        evaluate_fn=_evaluate_fn,
    )

    final_model = make_model(context_dim=context_dim)
    final_model.load_state_dict(result.arrays.to_torch_state_dict(), strict=True)
    final_model = final_model.to(device)

    eval_out = evaluate_with_gate(
        model=final_model,
        val_loader=server_eval["loaders"]["val"],
        test_loader=server_eval["loaders"]["test"],
        device=device,
        unknown_idx=cfg.unknown_gate.unknown_class_index,
        grid_size=cfg.unknown_gate.threshold_grid_size,
    )

    params = count_parameters(final_model)
    edge = evaluate_edge_constraints(
        model=final_model,
        seq_sample=server_eval["normed"]["test"]["X_seq_n"][0],
        ctx_sample=server_eval["normed"]["test"]["X_context_n"][0],
        params=params,
    )

    domain_metrics = {
        "domain": domain,
        "unknown_threshold": eval_out["unknown_threshold"],
        "raw_metrics": eval_out["raw_metrics"],
        "gated_metrics": eval_out["gated_metrics"],
        "edge": edge,
        "acceptance_targets": V2_ACCEPTANCE_TARGETS,
        "edge_targets": V2_EDGE_TARGETS,
    }

    paths = _domain_output_paths(cfg, domain)
    with paths["metrics"].open("w", encoding="utf-8") as f:
        json.dump(to_builtin(domain_metrics), f, indent=2)

    with paths["threshold"].open("w", encoding="utf-8") as f:
        json.dump({"unknown_threshold": float(eval_out["unknown_threshold"])}, f, indent=2)

    np.save(paths["cm"], np.array(eval_out["gated_metrics"]["confusion_matrix"], dtype=np.int64))
    torch.save(final_model.state_dict(), paths["checkpoint"])

    with paths["contract"].open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": cfg.schema_version,
                "run_name": cfg.artifacts.run_name,
                "domain": domain,
                "mode": cfg.mode,
            },
            f,
            indent=2,
        )

    return {
        "domain": domain,
        "domain_metrics": domain_metrics,
        "summary_row": format_run_result_for_summary(domain, eval_out, edge),
        "paths": {k: str(v) for k, v in paths.items()},
    }
