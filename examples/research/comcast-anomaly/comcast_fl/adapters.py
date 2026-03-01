"""Runtime adapters for simulation and deployment execution."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from flwr.simulation import run_simulation
import numpy as np
import torch

from .app_state import build_run_config_payload, set_active_experiment
from .config import ExperimentConfig
from .constants import V2_SIGNAL_DOMAINS
from .federated_core import build_client_bundle, build_server_eval_bundle, get_device, make_model
from .model import count_parameters
from .training import evaluate_edge_constraints, evaluate_with_gate, train_local_epoch
from .utils import ensure_dir, to_builtin


APP_DIR = Path(__file__).resolve().parents[1]


def _domain_metrics_path(cfg: ExperimentConfig, domain: str) -> Path:
    return Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / domain / "metrics.json"


def _read_domain_metrics(cfg: ExperimentConfig, domain: str) -> dict:
    path = _domain_metrics_path(cfg, domain)
    if not path.exists():
        raise FileNotFoundError(f"Expected domain metrics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_domain_simulation(domain: str, cfg: ExperimentConfig) -> dict:
    """Run one domain in simulation mode with variable number of clients."""
    if domain not in V2_SIGNAL_DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")

    set_active_experiment(cfg, domain)

    from .flower_client_app import app as client_app
    from .flower_server_app import app as server_app

    ray_available = importlib.util.find_spec("ray") is not None
    if ray_available:
        try:
            run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=int(cfg.federation.num_clients),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = str(exc).lower()
            if "ray" not in msg and "backend" not in msg:
                raise
            _run_domain_simulation_fallback(domain=domain, cfg=cfg)
    else:
        _run_domain_simulation_fallback(domain=domain, cfg=cfg)

    domain_metrics = _read_domain_metrics(cfg, domain)
    return {
        "domain": domain,
        "mode": "simulation",
        "domain_metrics": domain_metrics,
    }


def _make_run_config_toml(cfg: ExperimentConfig, domain: str) -> str:
    payload = build_run_config_payload(cfg, domain)
    lines = []
    for k, v in payload.items():
        lines.append(f"{k} = {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def run_domain_deployment(domain: str, cfg: ExperimentConfig) -> dict:
    """Run one domain through `flwr run` against a configured SuperLink/Federation."""
    if domain not in V2_SIGNAL_DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")

    if not cfg.deployment.superlink:
        raise ValueError("deployment.superlink must be set for deployment mode")

    toml_text = _make_run_config_toml(cfg, domain)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
        tmp.write(toml_text)
        tmp_path = Path(tmp.name)

    cmd = [
        "flwr",
        "run",
        str(APP_DIR),
        str(cfg.deployment.superlink),
        "--run-config",
        str(tmp_path),
    ]
    if cfg.deployment.federation:
        cmd.extend(["--federation", str(cfg.deployment.federation)])
    if cfg.deployment.stream_logs:
        cmd.append("--stream")

    proc = subprocess.run(
        cmd,
        cwd=str(APP_DIR),
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Deployment run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    domain_metrics = _read_domain_metrics(cfg, domain)
    return {
        "domain": domain,
        "mode": "deployment",
        "domain_metrics": domain_metrics,
        "stdout": proc.stdout,
    }


def write_run_summary(results: dict[str, dict], out_path: str, cfg: ExperimentConfig) -> None:
    """Write cross-domain summary JSON and comparison CSV."""
    summary_path = Path(out_path)
    ensure_dir(summary_path.parent)

    comparison_rows = []
    for domain in cfg.domains:
        dm = results[domain]["domain_metrics"]
        row = {
            "domain": domain,
            "gated_macro_f1": dm["gated_metrics"]["macro_f1"],
            "gated_event_peak_macro_f1": dm["gated_metrics"]["event_peak_macro_f1"],
            "gated_impulse_f1": dm["gated_metrics"]["impulse_f1"],
            "gated_unknown_f1": dm["gated_metrics"]["unknown_f1"],
            "anomaly_auroc": dm["gated_metrics"]["anomaly_auroc"],
            "unknown_threshold": dm["unknown_threshold"],
            "params": dm["edge"]["params"],
            "p95_latency_ms_cpu_proxy": dm["edge"]["p95_latency_ms_cpu_proxy"],
            "quantization_ready": dm["edge"]["quantization_ready"],
            "pass_edge_gate": dm["edge"]["pass_edge_gate"],
        }
        comparison_rows.append(row)

    comparison_csv = summary_path.parent / "comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "config": cfg.to_dict(),
        "domains": {k: to_builtin(v["domain_metrics"]) for k, v in results.items()},
        "comparison": to_builtin(comparison_rows),
        "comparison_csv": str(comparison_csv),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(summary), f, indent=2)


def _domain_output_paths(cfg: ExperimentConfig, domain: str) -> dict[str, Path]:
    root = Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / domain
    ensure_dir(root)
    return {
        "root": root,
        "metrics": root / "metrics.json",
        "checkpoint": root / "checkpoint_best.pt",
        "cm": root / "confusion_matrix.npy",
        "threshold": root / "threshold.json",
    }


def _aggregate_state_dicts(state_dicts: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    out: dict[str, torch.Tensor] = {}
    for k in state_dicts[0].keys():
        acc = None
        for sd, w in zip(state_dicts, weights):
            weighted = sd[k].detach().cpu() * (float(w) / total)
            acc = weighted if acc is None else acc + weighted
        out[k] = acc  # type: ignore[assignment]
    return out


def _run_domain_simulation_fallback(domain: str, cfg: ExperimentConfig) -> None:
    """Fallback local FedAvg loop used when Flower Simulation backend is unavailable."""
    device = get_device()
    num_clients = int(cfg.federation.num_clients)
    frac = float(cfg.federation.fraction_train)
    min_train = int(cfg.federation.min_train_nodes)
    local_epochs = int(cfg.local_training.local_epochs)
    lr = float(cfg.local_training.lr)
    wd = float(cfg.local_training.weight_decay)

    context_dim = 12
    global_model = make_model(context_dim=context_dim)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    rng = np.random.default_rng(int(cfg.seed))
    all_clients = [str(i) for i in range(num_clients)]

    for _round in range(1, int(cfg.federation.num_rounds) + 1):
        sample_size = max(min_train, int(math.ceil(frac * num_clients)))
        sampled = rng.choice(all_clients, size=sample_size, replace=False).tolist()

        local_states = []
        local_weights = []
        for cid in sampled:
            bundle = build_client_bundle(client_id=cid, domain=domain, cfg=cfg)
            model = make_model(context_dim=bundle["context_dim"])
            model.load_state_dict(global_state, strict=True)
            model = model.to(device)
            _, n_seen = train_local_epoch(
                model=model,
                loader=bundle["loaders"]["train"],
                device=device,
                lr=lr,
                weight_decay=wd,
                local_epochs=local_epochs,
            )
            local_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            local_weights.append(int(n_seen))

        global_state = _aggregate_state_dicts(local_states, local_weights)

    final_model = make_model(context_dim=context_dim)
    final_model.load_state_dict(global_state, strict=True)
    final_model = final_model.to(device)

    server_eval = build_server_eval_bundle(domain=domain, cfg=cfg)
    eval_out = evaluate_with_gate(
        model=final_model,
        val_loader=server_eval["loaders"]["val"],
        test_loader=server_eval["loaders"]["test"],
        device=device,
        unknown_idx=cfg.unknown_gate.unknown_class_index,
        grid_size=cfg.unknown_gate.threshold_grid_size,
    )
    edge = evaluate_edge_constraints(
        model=final_model,
        seq_sample=server_eval["normed"]["test"]["X_seq_n"][0],
        ctx_sample=server_eval["normed"]["test"]["X_context_n"][0],
        params=count_parameters(final_model),
    )
    metrics = {
        "domain": domain,
        "unknown_threshold": eval_out["unknown_threshold"],
        "raw_metrics": eval_out["raw_metrics"],
        "gated_metrics": eval_out["gated_metrics"],
        "edge": edge,
    }

    paths = _domain_output_paths(cfg, domain)
    with paths["metrics"].open("w", encoding="utf-8") as f:
        json.dump(to_builtin(metrics), f, indent=2)
    with paths["threshold"].open("w", encoding="utf-8") as f:
        json.dump({"unknown_threshold": float(eval_out["unknown_threshold"])}, f, indent=2)
    np.save(paths["cm"], np.array(eval_out["gated_metrics"]["confusion_matrix"], dtype=np.int64))
    torch.save(final_model.state_dict(), paths["checkpoint"])
