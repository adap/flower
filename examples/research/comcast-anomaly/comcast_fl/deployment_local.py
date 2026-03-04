"""Managed local multi-process deployment runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any

from .config import ExperimentConfig
from .ui_hooks import UiHookSink, emit_hook
from .utils import ensure_dir


@dataclass(slots=True)
class ManagedRuntimeHandle:
    """Runtime process handles and resolved connection metadata."""

    connection_name: str
    num_supernodes: int
    control_api_addr: str
    fleet_api_addr: str
    serverappio_api_addr: str
    clientappio_addrs: list[str]
    runtime_dir: Path
    logs_dir: Path
    flwr_home: Path
    env: dict[str, str]
    superlink_proc: subprocess.Popen[Any]
    supernode_procs: list[subprocess.Popen[Any]]
    runtime_state_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_host_port(addr: str) -> tuple[str, int]:
    host, port = addr.rsplit(":", maxsplit=1)
    return host, int(port)


def _reserve_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _wait_for_tcp(addr: str, timeout_sec: float) -> None:
    host, port = _split_host_port(addr)
    deadline = time.monotonic() + float(timeout_sec)
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.4):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for TCP endpoint {addr}: {last_err}")


def _start_logged_process(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
    cwd: Path | None = None,
) -> subprocess.Popen[Any]:
    ensure_dir(log_path.parent)
    with log_path.open("ab") as log_file:
        return subprocess.Popen(  # pylint: disable=consider-using-with
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            start_new_session=True,
        )


def _write_flwr_home_config(
    flwr_home: Path,
    connection_name: str,
    control_api_addr: str,
    insecure: bool,
) -> Path:
    ensure_dir(flwr_home)
    cfg_path = flwr_home / "config.toml"
    text = "\n".join(
        [
            "[superlink]",
            f'default = "{connection_name}"',
            "",
            f"[superlink.{connection_name}]",
            f'address = "{control_api_addr}"',
            f"insecure = {'true' if insecure else 'false'}",
            "",
        ]
    )
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def _make_superlink_cmd(
    control_api_addr: str,
    fleet_api_addr: str,
    serverappio_api_addr: str,
    database: str,
    storage_dir: Path,
) -> list[str]:
    return [
        "flower-superlink",
        "--insecure",
        "--control-api-address",
        control_api_addr,
        "--fleet-api-address",
        fleet_api_addr,
        "--serverappio-api-address",
        serverappio_api_addr,
        "--database",
        database,
        "--storage-dir",
        str(storage_dir),
    ]


def _make_supernode_cmd(
    fleet_api_addr: str,
    clientappio_api_addr: str,
    partition_id: int,
    num_partitions: int,
) -> list[str]:
    return [
        "flower-supernode",
        "--insecure",
        "--superlink",
        fleet_api_addr,
        "--clientappio-api-address",
        clientappio_api_addr,
        "--max-retries",
        "0",
        "--node-config",
        (
            f"partition-id={partition_id} "
            f"num-partitions={num_partitions} "
            f"client-id={partition_id}"
        ),
    ]


def _resolve_num_supernodes(cfg: ExperimentConfig) -> int:
    if cfg.deployment.local_num_supernodes is not None:
        return int(cfg.deployment.local_num_supernodes)
    return int(cfg.federation.num_clients)


def _resolve_runtime_dir(cfg: ExperimentConfig) -> Path:
    if cfg.deployment.local_runtime_dir:
        return Path(cfg.deployment.local_runtime_dir)
    return Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / "deployment_runtime"


def _cleanup_runtime_payloads(runtime_dir: Path) -> dict[str, bool]:
    """Remove heavy nested payloads that are not needed between runs."""
    flwr_apps_dir = runtime_dir / "flwr_home" / "apps"
    storage_dir = runtime_dir / "storage"
    removed_apps = False
    removed_storage = False
    if flwr_apps_dir.exists():
        shutil.rmtree(flwr_apps_dir, ignore_errors=True)
        removed_apps = True
    if storage_dir.exists():
        shutil.rmtree(storage_dir, ignore_errors=True)
        removed_storage = True
    return {"removed_flwr_apps": removed_apps, "removed_storage": removed_storage}


def _write_runtime_state(handle: ManagedRuntimeHandle, extra: dict[str, Any] | None = None) -> None:
    state = {
        "connection_name": handle.connection_name,
        "num_supernodes": handle.num_supernodes,
        "control_api_addr": handle.control_api_addr,
        "fleet_api_addr": handle.fleet_api_addr,
        "serverappio_api_addr": handle.serverappio_api_addr,
        "clientappio_addrs": handle.clientappio_addrs,
        "runtime_dir": str(handle.runtime_dir),
        "logs_dir": str(handle.logs_dir),
        "flwr_home": str(handle.flwr_home),
        "processes": {
            "superlink_pid": handle.superlink_proc.pid,
            "supernode_pids": [p.pid for p in handle.supernode_procs],
        },
        "updated_at": _utc_now(),
    }
    if extra:
        state.update(extra)
    handle.runtime_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def start_managed_local_runtime(
    cfg: ExperimentConfig,
    hook_sink: UiHookSink | None = None,
) -> ManagedRuntimeHandle:
    """Start SuperLink + SuperNodes locally for deployment-mode runs."""
    num_supernodes = _resolve_num_supernodes(cfg)
    runtime_dir = _resolve_runtime_dir(cfg)
    cleanup_info = _cleanup_runtime_payloads(runtime_dir)
    logs_dir = runtime_dir / "logs"
    storage_dir = runtime_dir / "storage"
    flwr_home = runtime_dir / "flwr_home"
    state_path = runtime_dir / "runtime_state.json"
    ensure_dir(logs_dir)
    ensure_dir(storage_dir)
    ensure_dir(flwr_home)

    control_api_addr = f"127.0.0.1:{_reserve_free_port()}"
    fleet_api_addr = f"127.0.0.1:{_reserve_free_port()}"
    serverappio_api_addr = f"127.0.0.1:{_reserve_free_port()}"
    clientappio_addrs = [f"127.0.0.1:{_reserve_free_port()}" for _ in range(num_supernodes)]

    _write_flwr_home_config(
        flwr_home=flwr_home,
        connection_name=cfg.deployment.connection_name,
        control_api_addr=control_api_addr,
        insecure=bool(cfg.deployment.local_insecure),
    )

    env = os.environ.copy()
    env["FLWR_HOME"] = str(flwr_home)
    env["COMCAST_FL_RUNTIME_LOG_DIR"] = str(logs_dir)

    superlink_cmd = _make_superlink_cmd(
        control_api_addr=control_api_addr,
        fleet_api_addr=fleet_api_addr,
        serverappio_api_addr=serverappio_api_addr,
        database=cfg.deployment.local_database,
        storage_dir=storage_dir,
    )
    superlink_proc = _start_logged_process(
        cmd=superlink_cmd,
        log_path=logs_dir / "superlink.log",
        env=env,
    )

    supernode_procs: list[subprocess.Popen[Any]] = []
    handle: ManagedRuntimeHandle | None = None
    try:
        _wait_for_tcp(control_api_addr, timeout_sec=float(cfg.deployment.startup_timeout_sec))
        _wait_for_tcp(fleet_api_addr, timeout_sec=float(cfg.deployment.startup_timeout_sec))
        _wait_for_tcp(serverappio_api_addr, timeout_sec=float(cfg.deployment.startup_timeout_sec))

        for idx, client_addr in enumerate(clientappio_addrs):
            cmd = _make_supernode_cmd(
                fleet_api_addr=fleet_api_addr,
                clientappio_api_addr=client_addr,
                partition_id=idx,
                num_partitions=num_supernodes,
            )
            proc = _start_logged_process(
                cmd=cmd,
                log_path=logs_dir / f"supernode_{idx}.log",
                env=env,
            )
            supernode_procs.append(proc)

        for client_addr in clientappio_addrs:
            _wait_for_tcp(client_addr, timeout_sec=float(cfg.deployment.startup_timeout_sec))

        # SuperNode registration window to let all nodes become available to the run.
        time.sleep(min(3.0, max(1.0, float(cfg.deployment.poll_interval_sec))))

        handle = ManagedRuntimeHandle(
            connection_name=cfg.deployment.connection_name,
            num_supernodes=num_supernodes,
            control_api_addr=control_api_addr,
            fleet_api_addr=fleet_api_addr,
            serverappio_api_addr=serverappio_api_addr,
            clientappio_addrs=clientappio_addrs,
            runtime_dir=runtime_dir,
            logs_dir=logs_dir,
            flwr_home=flwr_home,
            env=env,
            superlink_proc=superlink_proc,
            supernode_procs=supernode_procs,
            runtime_state_path=state_path,
        )
        _write_runtime_state(handle, extra={"started_at": _utc_now(), "cleanup": cleanup_info})
        emit_hook(
            hook_sink,
            event_type="runtime.started",
            payload=json.loads(handle.runtime_state_path.read_text(encoding="utf-8")),
            run_name=cfg.artifacts.run_name,
            domain=None,
        )
        return handle
    except Exception:
        if handle is None:
            handle = ManagedRuntimeHandle(
                connection_name=cfg.deployment.connection_name,
                num_supernodes=num_supernodes,
                control_api_addr=control_api_addr,
                fleet_api_addr=fleet_api_addr,
                serverappio_api_addr=serverappio_api_addr,
                clientappio_addrs=clientappio_addrs,
                runtime_dir=runtime_dir,
                logs_dir=logs_dir,
                flwr_home=flwr_home,
                env=env,
                superlink_proc=superlink_proc,
                supernode_procs=supernode_procs,
                runtime_state_path=state_path,
            )
        stop_managed_local_runtime(
            handle,
            quiet=True,
            shutdown_grace_sec=cfg.deployment.shutdown_grace_sec,
            hook_sink=hook_sink,
            run_name=cfg.artifacts.run_name,
        )
        raise


def _terminate_process(
    proc: subprocess.Popen[Any],
    shutdown_grace_sec: int,
) -> dict[str, Any]:
    exit_code = proc.poll()
    was_running = exit_code is None
    if exit_code is None:
        proc.terminate()
        try:
            proc.wait(timeout=float(shutdown_grace_sec))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
    return {
        "pid": proc.pid,
        "was_running": was_running,
        "exit_code": proc.poll(),
    }


def stop_managed_local_runtime(
    handle: ManagedRuntimeHandle,
    quiet: bool = False,
    shutdown_grace_sec: int | None = None,
    hook_sink: UiHookSink | None = None,
    run_name: str | None = None,
) -> None:
    """Stop managed local runtime processes and persist teardown details."""
    grace = int(shutdown_grace_sec if shutdown_grace_sec is not None else 5)
    supernode_results = [
        _terminate_process(proc, shutdown_grace_sec=grace) for proc in reversed(handle.supernode_procs)
    ]
    superlink_result = _terminate_process(handle.superlink_proc, shutdown_grace_sec=grace)

    cleanup_info = _cleanup_runtime_payloads(handle.runtime_dir)
    summary = {
        "stopped_at": _utc_now(),
        "cleanup": cleanup_info,
        "teardown": {
            "superlink": superlink_result,
            "supernodes": list(reversed(supernode_results)),
            "all_exited": superlink_result["exit_code"] is not None
            and all(r["exit_code"] is not None for r in supernode_results),
        },
    }
    _write_runtime_state(handle, extra=summary)
    emit_hook(
        hook_sink,
        event_type="runtime.stopped",
        payload=json.loads(handle.runtime_state_path.read_text(encoding="utf-8")),
        run_name=run_name,
        domain=None,
    )

    if not quiet:
        print(
            "Managed local runtime teardown:",
            json.dumps(summary["teardown"], sort_keys=True),
        )
