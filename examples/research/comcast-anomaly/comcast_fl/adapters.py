"""Runtime adapters for simulation and deployment execution."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from flwr.simulation import run_simulation
import numpy as np
import torch

from .app_state import build_run_config_payload, set_active_experiment
from .config import ExperimentConfig
from .constants import V2_SIGNAL_DOMAINS
from .deployment_azure_ssh import (
    ManagedAzureRuntimeHandle,
    collect_remote_artifacts,
    make_remote_run_config_toml,
    start_managed_azure_runtime,
    stop_managed_azure_runtime,
    submit_run_and_wait_remote,
)
from .deployment_local import (
    ManagedRuntimeHandle,
    start_managed_local_runtime,
    stop_managed_local_runtime,
)
from .federated_core import build_client_bundle, build_server_eval_bundle, get_device, make_model
from .model import count_parameters
from .training import evaluate_edge_constraints, evaluate_with_gate, train_local_epoch
from .ui_hooks import UiHookSink, emit_hook
from .utils import ensure_dir, to_builtin


APP_DIR = Path(__file__).resolve().parents[1]


def _domain_metrics_path(cfg: ExperimentConfig, domain: str) -> Path:
    return Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / domain / "metrics.json"


def _read_domain_metrics(cfg: ExperimentConfig, domain: str) -> dict:
    path = _domain_metrics_path(cfg, domain)
    if not path.exists():
        raise FileNotFoundError(f"Expected domain metrics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_domain_simulation(
    domain: str,
    cfg: ExperimentConfig,
    hook_sink: UiHookSink | None = None,
) -> dict:
    """Run one domain in simulation mode with variable number of clients."""
    if domain not in V2_SIGNAL_DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")

    emit_hook(
        hook_sink,
        event_type="domain.started",
        payload={"mode": "simulation"},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )
    emit_hook(
        hook_sink,
        event_type="run.started",
        payload={"mode": "simulation", "domain": domain},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )
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
            _run_domain_simulation_fallback(domain=domain, cfg=cfg, hook_sink=hook_sink)
    else:
        _run_domain_simulation_fallback(domain=domain, cfg=cfg, hook_sink=hook_sink)

    domain_metrics = _read_domain_metrics(cfg, domain)
    emit_hook(
        hook_sink,
        event_type="metrics.updated",
        payload={"metrics": domain_metrics, "domain": domain},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )
    emit_hook(
        hook_sink,
        event_type="domain.completed",
        payload={"status": "finished:completed"},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )
    emit_hook(
        hook_sink,
        event_type="run.completed",
        payload={"mode": "simulation", "domain": domain},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )
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


def submit_run_and_wait(
    app_dir: Path,
    connection_name: str,
    run_config_toml: Path,
    timeout_sec: int,
    poll_sec: float,
    env: dict[str, str],
    run_name: str | None = None,
    domain: str | None = None,
    hook_sink: UiHookSink | None = None,
) -> dict:
    """Submit a Flower run and wait for completion/failure with timeout."""
    cmd = [
        "flwr",
        "run",
        str(app_dir),
        str(connection_name),
        "--run-config",
        str(run_config_toml),
        "--format",
        "json",
    ]
    federation_override = env.get("COMCAST_FL_FEDERATION", "").strip()
    if federation_override:
        cmd.extend(["--federation", federation_override])

    proc = subprocess.run(
        cmd,
        cwd=str(app_dir),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Deployment run submission failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
            f"Runtime logs: {env.get('COMCAST_FL_RUNTIME_LOG_DIR', '(not set)')}"
        )

    run_payload = json.loads(proc.stdout)
    if run_payload.get("success") is False:
        err = str(run_payload.get("error-message", "")).strip()
        raise RuntimeError(
            "Deployment run submission reported failure.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error: {err or '(no error-message provided)'}\n"
            f"Runtime logs: {env.get('COMCAST_FL_RUNTIME_LOG_DIR', '(not set)')}"
        )
    run_id = run_payload.get("run-id")
    if run_id is None:
        raise RuntimeError(f"Could not parse run-id from `flwr run` output: {proc.stdout}")
    run_id = int(run_id)
    emit_hook(
        hook_sink,
        event_type="run.started",
        payload={"run_id": run_id, "connection_name": connection_name},
        run_name=run_name,
        domain=domain,
    )

    deadline = time.monotonic() + float(timeout_sec)
    last_status = "unknown"
    last_runs_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        ls_cmd = [
            "flwr",
            "ls",
            str(connection_name),
            "--runs",
            "--format",
            "json",
        ]
        ls_proc = subprocess.run(
            ls_cmd,
            cwd=str(app_dir),
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if ls_proc.returncode != 0:
            raise RuntimeError(
                "Failed while polling run status.\n"
                f"Command: {' '.join(ls_cmd)}\n"
                f"STDOUT:\n{ls_proc.stdout}\n"
                f"STDERR:\n{ls_proc.stderr}\n"
                f"Runtime logs: {env.get('COMCAST_FL_RUNTIME_LOG_DIR', '(not set)')}"
            )
        last_runs_payload = json.loads(ls_proc.stdout)
        runs = last_runs_payload.get("runs", [])
        found = next((r for r in runs if int(r.get("run-id", -1)) == run_id), None)
        if found is not None:
            last_status = str(found.get("status", "unknown"))
            emit_hook(
                hook_sink,
                event_type="run.status",
                payload={"run_id": run_id, "status": last_status},
                run_name=run_name,
                domain=domain,
            )
            if last_status.startswith("finished:"):
                if last_status == "finished:completed":
                    emit_hook(
                        hook_sink,
                        event_type="run.completed",
                        payload={"run_id": run_id, "status": last_status},
                        run_name=run_name,
                        domain=domain,
                    )
                    return {
                        "run_id": run_id,
                        "status": last_status,
                        "run_payload": run_payload,
                        "status_payload": last_runs_payload,
                    }
                emit_hook(
                    hook_sink,
                    event_type="run.failed",
                    payload={"run_id": run_id, "status": last_status},
                    run_name=run_name,
                    domain=domain,
                )
                raise RuntimeError(
                    "Run finished in non-success state.\n"
                    f"run_id={run_id}, status={last_status}\n"
                    f"Runtime logs: {env.get('COMCAST_FL_RUNTIME_LOG_DIR', '(not set)')}\n"
                    f"Status payload: {json.dumps(last_runs_payload)}"
                )
        time.sleep(float(poll_sec))

    emit_hook(
        hook_sink,
        event_type="run.timeout",
        payload={"run_id": run_id, "status": last_status, "timeout_sec": timeout_sec},
        run_name=run_name,
        domain=domain,
    )
    raise TimeoutError(
        "Timed out waiting for run completion.\n"
        f"run_id={run_id}, last_status={last_status}, timeout_sec={timeout_sec}\n"
        f"Runtime logs: {env.get('COMCAST_FL_RUNTIME_LOG_DIR', '(not set)')}\n"
        f"Last status payload: {json.dumps(last_runs_payload)}"
    )


def run_domain_deployment(
    domain: str,
    cfg: ExperimentConfig,
    runtime: ManagedRuntimeHandle | ManagedAzureRuntimeHandle | None = None,
    hook_sink: UiHookSink | None = None,
) -> dict:
    """Run one domain in deployment mode."""
    if domain not in V2_SIGNAL_DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")
    emit_hook(
        hook_sink,
        event_type="domain.started",
        payload={"mode": "deployment"},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )

    runtime_owned = False
    runtime_handle: ManagedRuntimeHandle | ManagedAzureRuntimeHandle | None = runtime
    if cfg.deployment.launch_mode == "managed_local":
        if runtime_handle is None:
            runtime_owned = True
            runtime_handle = start_managed_local_runtime(cfg, hook_sink=hook_sink)
        if not isinstance(runtime_handle, ManagedRuntimeHandle):
            raise TypeError("Expected ManagedRuntimeHandle for managed_local deployment")
        connection_name = runtime_handle.connection_name
        env = runtime_handle.env
    elif cfg.deployment.launch_mode == "managed_azure_ssh":
        if runtime_handle is None:
            runtime_owned = True
            runtime_handle = start_managed_azure_runtime(cfg, hook_sink=hook_sink)
        if not isinstance(runtime_handle, ManagedAzureRuntimeHandle):
            raise TypeError("Expected ManagedAzureRuntimeHandle for managed_azure_ssh deployment")
        connection_name = runtime_handle.connection_name
        env = {}
    else:
        if not cfg.deployment.superlink:
            raise ValueError("deployment.superlink must be set for external deployment mode")
        connection_name = str(cfg.deployment.superlink)
        env = os.environ.copy()

    if cfg.deployment.launch_mode == "managed_azure_ssh":
        assert isinstance(runtime_handle, ManagedAzureRuntimeHandle)
        toml_text = make_remote_run_config_toml(
            cfg=cfg,
            domain=domain,
            remote_output_root=runtime_handle.remote_artifacts_root,
        )
        tmp_path: Path | None = None
    else:
        toml_text = _make_run_config_toml(cfg, domain)
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as tmp:
            tmp.write(toml_text)
            tmp_path = Path(tmp.name)

    if cfg.deployment.federation:
        env = dict(env)
        env["COMCAST_FL_FEDERATION"] = str(cfg.deployment.federation)

    try:
        if cfg.deployment.launch_mode == "managed_azure_ssh":
            assert isinstance(runtime_handle, ManagedAzureRuntimeHandle)
            run_info = submit_run_and_wait_remote(
                handle=runtime_handle,
                domain=domain,
                run_config_toml_text=toml_text,
                cfg=cfg,
                hook_sink=hook_sink,
            )
            collect_remote_artifacts(runtime_handle, cfg)
        else:
            assert tmp_path is not None
            run_info = submit_run_and_wait(
                app_dir=APP_DIR,
                connection_name=connection_name,
                run_config_toml=tmp_path,
                timeout_sec=int(cfg.deployment.run_timeout_sec),
                poll_sec=float(cfg.deployment.poll_interval_sec),
                env=env,
                run_name=cfg.artifacts.run_name,
                domain=domain,
                hook_sink=hook_sink,
            )
        domain_metrics = _read_domain_metrics(cfg, domain)
        emit_hook(
            hook_sink,
            event_type="metrics.updated",
            payload={"metrics": domain_metrics, "domain": domain},
            run_name=cfg.artifacts.run_name,
            domain=domain,
        )
        emit_hook(
            hook_sink,
            event_type="domain.completed",
            payload={"status": "finished:completed"},
            run_name=cfg.artifacts.run_name,
            domain=domain,
        )
        return {
            "domain": domain,
            "mode": "deployment",
            "domain_metrics": domain_metrics,
            "run": run_info,
        }
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        if runtime_owned and runtime_handle is not None:
            if isinstance(runtime_handle, ManagedRuntimeHandle):
                stop_managed_local_runtime(
                    runtime_handle,
                    shutdown_grace_sec=int(cfg.deployment.shutdown_grace_sec),
                    hook_sink=hook_sink,
                    run_name=cfg.artifacts.run_name,
                )
            elif isinstance(runtime_handle, ManagedAzureRuntimeHandle):
                stop_managed_azure_runtime(runtime_handle, quiet=False)


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


def _run_domain_simulation_fallback(
    domain: str,
    cfg: ExperimentConfig,
    hook_sink: UiHookSink | None = None,
) -> None:
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
        emit_hook(
            hook_sink,
            event_type="run.status",
            payload={"status": f"running:round:{_round}", "sampled_clients": sampled},
            run_name=cfg.artifacts.run_name,
            domain=domain,
        )

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
