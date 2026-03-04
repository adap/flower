"""Managed Azure SSH deployment runtime helpers (TLS + SuperNode auth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Any

from .config import AzureVmSpec, ExperimentConfig, loads_config_json
from .ui_hooks import UiHookSink, emit_hook
from .utils import ensure_dir


APP_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ManagedAzureRuntimeHandle:
    """Resolved runtime metadata for managed Azure SSH mode."""

    connection_name: str
    run_name: str
    local_runtime_dir: Path
    local_logs_dir: Path
    runtime_state_path: Path
    local_artifacts_root: Path
    remote_workspace_dir: str
    remote_app_dir: str
    remote_runtime_dir: str
    remote_artifacts_root: str
    control_vm: AzureVmSpec
    superlink_vm: AzureVmSpec
    vm_by_name: dict[str, AzureVmSpec]
    control_api_addr: str
    fleet_api_addr: str
    serverappio_api_addr: str
    control_flwr_home: str
    tls_remote_paths: dict[str, str]
    superlink_pid_file: str
    supernode_entries: list[dict[str, Any]]
    startup_timeout_sec: int
    poll_interval_sec: float
    domain_run_timeout_sec: int
    teardown_grace_sec: int
    ssh_connect_timeout_sec: int
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _ssh_base(vm: AzureVmSpec, connect_timeout_sec: int) -> list[str]:
    return [
        "ssh",
        "-i",
        vm.ssh_key_path,
        "-p",
        str(vm.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={int(connect_timeout_sec)}",
        f"{vm.ssh_user}@{vm.host}",
    ]


def _scp_base(vm: AzureVmSpec, connect_timeout_sec: int) -> list[str]:
    return [
        "scp",
        "-i",
        vm.ssh_key_path,
        "-P",
        str(vm.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={int(connect_timeout_sec)}",
    ]


def _run_remote_command(
    vm: AzureVmSpec,
    remote_cmd: str,
    connect_timeout_sec: int,
    timeout_sec: float | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = _ssh_base(vm, connect_timeout_sec) + ["bash", "-lc", remote_cmd]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_sec,
        env=env,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            "Remote command failed.\n"
            f"vm={vm.name} host={vm.host}\n"
            f"command={remote_cmd}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _copy_to_remote(
    vm: AzureVmSpec,
    local_path: Path,
    remote_path: str,
    connect_timeout_sec: int,
) -> None:
    _run_remote_command(
        vm=vm,
        remote_cmd=f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}",
        connect_timeout_sec=connect_timeout_sec,
        timeout_sec=30.0,
        check=True,
    )
    cmd = _scp_base(vm, connect_timeout_sec) + [str(local_path), f"{vm.ssh_user}@{vm.host}:{remote_path}"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "SCP to remote failed.\n"
            f"vm={vm.name} host={vm.host}\n"
            f"local={local_path}\n"
            f"remote={remote_path}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _copy_from_remote(
    vm: AzureVmSpec,
    remote_path: str,
    local_path: Path,
    connect_timeout_sec: int,
) -> None:
    ensure_dir(local_path.parent)
    cmd = _scp_base(vm, connect_timeout_sec) + [
        "-r",
        f"{vm.ssh_user}@{vm.host}:{remote_path}",
        str(local_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "SCP from remote failed.\n"
            f"vm={vm.name} host={vm.host}\n"
            f"remote={remote_path}\n"
            f"local={local_path}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _prepare_app_archive(app_dir: Path) -> Path:
    """Create a small app archive with a strict allowlist."""
    allow = [
        "pyproject.toml",
        "README_FL.md",
        "comcast_fl",
        "configs",
    ]
    tmp_dir = Path(tempfile.mkdtemp(prefix="comcast-azure-app-"))
    archive_path = tmp_dir / "comcast-anomaly-app.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as tf:
        for rel in allow:
            src = app_dir / rel
            if not src.exists():
                continue
            tf.add(src, arcname=rel, recursive=True)
    return archive_path


def _validate_remote_workspace_dir(remote_workspace_dir: str) -> None:
    """Defensive runtime guard for destructive remote path operations."""
    _require(remote_workspace_dir.startswith("/"), "remote_workspace_dir must be absolute")
    _require(remote_workspace_dir != "/", "remote_workspace_dir cannot be root '/'")
    _require(".." not in remote_workspace_dir.split("/"), "remote_workspace_dir cannot contain '..'")
    _require(
        not any(ch in remote_workspace_dir for ch in ["*", "?", "[", "]", "{", "}", ";", "\n", "\r"]),
        "remote_workspace_dir contains unsafe characters",
    )


def _reserve_remote_port(vm: AzureVmSpec, remote_python: str, connect_timeout_sec: int) -> int:
    cmd = (
        f"{shlex.quote(remote_python)} -c "
        + shlex.quote(
            "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); "
            "print(s.getsockname()[1]); s.close()"
        )
    )
    proc = _run_remote_command(
        vm=vm,
        remote_cmd=cmd,
        connect_timeout_sec=connect_timeout_sec,
        timeout_sec=20.0,
        check=True,
    )
    return int(proc.stdout.strip().splitlines()[-1])


def _build_partition_plan(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    azure = cfg.deployment.azure_ssh
    auth = cfg.deployment.supernode_auth
    _require(azure is not None, "deployment.azure_ssh missing")
    _require(auth is not None, "deployment.supernode_auth missing")

    entries: list[dict[str, Any]] = []
    idx = 0
    for vm in azure.vms:
        for _ in range(int(vm.supernodes_on_vm)):
            entries.append(
                {
                    "partition_id": idx,
                    "vm_name": vm.name,
                    "private_key_local_path": auth.private_key_local_paths[idx],
                    "public_key_local_path": auth.public_key_local_paths[idx],
                }
            )
            idx += 1
    return entries


def _make_superlink_cmd(
    control_api_addr: str,
    fleet_api_addr: str,
    serverappio_api_addr: str,
    ca_cert: str,
    server_cert: str,
    server_key: str,
    storage_dir: str,
    database: str,
) -> list[str]:
    return [
        "flower-superlink",
        "--control-api-address",
        control_api_addr,
        "--fleet-api-address",
        fleet_api_addr,
        "--serverappio-api-address",
        serverappio_api_addr,
        "--ssl-ca-certfile",
        ca_cert,
        "--ssl-certfile",
        server_cert,
        "--ssl-keyfile",
        server_key,
        "--enable-supernode-auth",
        "--database",
        database,
        "--storage-dir",
        storage_dir,
    ]


def _make_supernode_cmd(
    fleet_api_addr: str,
    root_cert: str,
    private_key: str,
    clientappio_api_addr: str,
    partition_id: int,
    num_partitions: int,
) -> list[str]:
    return [
        "flower-supernode",
        "--superlink",
        fleet_api_addr,
        "--root-certificates",
        root_cert,
        "--auth-supernode-private-key",
        private_key,
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


def _start_remote_process(
    vm: AzureVmSpec,
    cmd: list[str],
    pid_file: str,
    log_file: str,
    connect_timeout_sec: int,
) -> None:
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    remote_cmd = (
        f"mkdir -p {shlex.quote(str(Path(pid_file).parent))} {shlex.quote(str(Path(log_file).parent))} && "
        f"nohup {cmd_str} > {shlex.quote(log_file)} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(pid_file)}"
    )
    _run_remote_command(
        vm=vm,
        remote_cmd=remote_cmd,
        connect_timeout_sec=connect_timeout_sec,
        timeout_sec=20.0,
        check=True,
    )


def _wait_for_remote_tcp(
    vm: AzureVmSpec,
    target_host: str,
    target_port: int,
    remote_python: str,
    timeout_sec: int,
    connect_timeout_sec: int,
) -> None:
    deadline = time.monotonic() + float(timeout_sec)
    one_try = (
        f"{shlex.quote(remote_python)} -c "
        + shlex.quote(
            "import socket,sys; "
            f"s=socket.socket(); s.settimeout(0.4); "
            f"target=({target_host!r},{target_port}); "
            "ok=0\n"
            "try:\n"
            " s.connect(target); ok=1\n"
            "except OSError:\n"
            " ok=0\n"
            "finally:\n"
            " s.close()\n"
            "sys.exit(0 if ok else 1)"
        )
    )
    last_err = ""
    while time.monotonic() < deadline:
        proc = _run_remote_command(
            vm=vm,
            remote_cmd=one_try,
            connect_timeout_sec=connect_timeout_sec,
            timeout_sec=5.0,
            check=False,
        )
        if proc.returncode == 0:
            return
        last_err = proc.stderr.strip() or proc.stdout.strip()
        time.sleep(0.3)
    raise RuntimeError(
        f"Timed out waiting for remote TCP endpoint {target_host}:{target_port} from vm={vm.name}. Last error: {last_err}"
    )


def _write_local_runtime_state(
    handle: ManagedAzureRuntimeHandle,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "mode": "managed_azure_ssh",
        "run_name": handle.run_name,
        "connection_name": handle.connection_name,
        "control_api_addr": handle.control_api_addr,
        "fleet_api_addr": handle.fleet_api_addr,
        "serverappio_api_addr": handle.serverappio_api_addr,
        "local_runtime_dir": str(handle.local_runtime_dir),
        "local_logs_dir": str(handle.local_logs_dir),
        "local_artifacts_root": str(handle.local_artifacts_root),
        "remote_workspace_dir": handle.remote_workspace_dir,
        "remote_app_dir": handle.remote_app_dir,
        "remote_runtime_dir": handle.remote_runtime_dir,
        "remote_artifacts_root": handle.remote_artifacts_root,
        "control_vm": handle.control_vm.name,
        "superlink_vm": handle.superlink_vm.name,
        "control_flwr_home_remote": handle.control_flwr_home,
        "superlink_pid_file": handle.superlink_pid_file,
        "supernode_entries": handle.supernode_entries,
        "updated_at": _utc_now(),
    }
    if extra:
        state.update(extra)
    ensure_dir(handle.runtime_state_path.parent)
    handle.runtime_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _stop_remote_process(vm: AzureVmSpec, pid_file: str, grace_sec: int, connect_timeout_sec: int) -> dict[str, Any]:
    remote_cmd = (
        f"if [ -f {shlex.quote(pid_file)} ]; then "
        f"pid=$(cat {shlex.quote(pid_file)}); "
        "if kill -0 \"$pid\" >/dev/null 2>&1; then kill \"$pid\" >/dev/null 2>&1 || true; fi; "
        f"sleep {int(grace_sec)}; "
        "if kill -0 \"$pid\" >/dev/null 2>&1; then kill -9 \"$pid\" >/dev/null 2>&1 || true; fi; "
        "if kill -0 \"$pid\" >/dev/null 2>&1; then echo RUNNING; else echo EXITED; fi; "
        "else echo MISSING_PID; fi"
    )
    proc = _run_remote_command(
        vm=vm,
        remote_cmd=remote_cmd,
        connect_timeout_sec=connect_timeout_sec,
        timeout_sec=float(max(10, grace_sec + 10)),
        check=False,
    )
    status = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "UNKNOWN"
    return {
        "vm": vm.name,
        "pid_file": pid_file,
        "status": status,
        "returncode": proc.returncode,
    }


def _best_effort_cleanup_on_start_failure(
    superlink_vm: AzureVmSpec,
    superlink_pid_file: str,
    started_supernode_pid_files: list[tuple[AzureVmSpec, str]],
    grace_sec: int,
    connect_timeout_sec: int,
) -> None:
    for vm, pid_file in reversed(started_supernode_pid_files):
        try:
            _ = _stop_remote_process(vm, pid_file, grace_sec=grace_sec, connect_timeout_sec=connect_timeout_sec)
        except Exception:
            pass
    if superlink_pid_file:
        try:
            _ = _stop_remote_process(
                superlink_vm,
                superlink_pid_file,
                grace_sec=grace_sec,
                connect_timeout_sec=connect_timeout_sec,
            )
        except Exception:
            pass


def _extract_run_id(stdout: str) -> int:
    payload = _parse_json_payload(stdout)
    rid = payload.get("run-id")
    if rid is None:
        raise RuntimeError(f"Could not parse run-id from remote `flwr run` output: {stdout}")
    return int(rid)


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError("Could not parse JSON payload", text, 0)


def _run_control_flwr(
    handle: ManagedAzureRuntimeHandle,
    args: list[str],
    timeout_sec: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    remote_cmd = (
        f"cd {shlex.quote(handle.remote_app_dir)} && "
        f"FLWR_HOME={shlex.quote(handle.control_flwr_home)} "
        + " ".join(shlex.quote(x) for x in args)
    )
    return _run_remote_command(
        vm=handle.control_vm,
        remote_cmd=remote_cmd,
        connect_timeout_sec=handle.ssh_connect_timeout_sec,
        timeout_sec=timeout_sec,
        check=check,
    )


def _wait_for_supernodes_online(
    handle: ManagedAzureRuntimeHandle,
    expected_online: int,
) -> None:
    deadline = time.monotonic() + float(handle.startup_timeout_sec)
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        proc = _run_control_flwr(
            handle=handle,
            args=["flwr", "supernode", "list", handle.connection_name, "--format", "json"],
            timeout_sec=20.0,
            check=False,
        )
        if proc.returncode == 0:
            try:
                payload = _parse_json_payload(proc.stdout)
            except json.JSONDecodeError:
                payload = {}
            last_payload = payload
            nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
            online = 0
            for n in nodes:
                status = str(n.get("status", "")).lower()
                if status == "online":
                    online += 1
            if online >= expected_online:
                return
        time.sleep(float(handle.poll_interval_sec))
    raise TimeoutError(
        "Timed out waiting for expected online SuperNodes.\n"
        f"expected_online={expected_online}\n"
        f"last_payload={json.dumps(last_payload)}"
    )


def start_managed_azure_runtime(
    cfg: ExperimentConfig,
    hook_sink: UiHookSink | None = None,
) -> ManagedAzureRuntimeHandle:
    """Start secure multi-VM Flower runtime on pre-provisioned Azure VMs."""
    azure = cfg.deployment.azure_ssh
    tls = cfg.deployment.tls
    auth = cfg.deployment.supernode_auth
    _require(azure is not None and tls is not None and auth is not None, "managed_azure_ssh config missing")

    local_runtime_dir = (
        Path(cfg.deployment.local_runtime_dir)
        if cfg.deployment.local_runtime_dir
        else Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / "deployment_runtime"
    )
    local_logs_dir = local_runtime_dir / "logs"
    local_artifacts_root = Path(cfg.artifacts.root_dir)
    runtime_state_path = local_runtime_dir / "runtime_state.json"
    ensure_dir(local_logs_dir)
    ensure_dir(local_artifacts_root)

    vm_by_name = {vm.name: vm for vm in azure.vms}
    control_vm = vm_by_name[azure.control_vm]
    superlink_vm = vm_by_name[azure.superlink_vm]
    _validate_remote_workspace_dir(azure.remote_workspace_dir)

    # Preflight and sync package.
    archive_path = _prepare_app_archive(APP_DIR)
    remote_runtime_dir = f"{azure.remote_workspace_dir.rstrip('/')}/runtime/{cfg.artifacts.run_name}"
    remote_app_dir = f"{azure.remote_workspace_dir.rstrip('/')}/app"
    remote_artifacts_root = f"{remote_runtime_dir}/artifacts/fl"

    handle: ManagedAzureRuntimeHandle | None = None
    superlink_pid_file = ""
    started_supernode_pid_files: list[tuple[AzureVmSpec, str]] = []
    try:
        for vm in azure.vms:
            _run_remote_command(
                vm=vm,
                remote_cmd=(
                    f"mkdir -p {shlex.quote(azure.remote_workspace_dir)} {shlex.quote(remote_runtime_dir)} "
                    f"{shlex.quote(remote_runtime_dir + '/logs')} {shlex.quote(remote_runtime_dir + '/pids')} "
                    f"{shlex.quote(remote_runtime_dir + '/run_configs')} {shlex.quote(remote_runtime_dir + '/state')}"
                ),
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
                timeout_sec=30.0,
                check=True,
            )
            _run_remote_command(
                vm=vm,
                remote_cmd=(
                    f"command -v {shlex.quote(azure.remote_python)} >/dev/null && "
                    "command -v flwr >/dev/null && "
                    "command -v flower-superlink >/dev/null && "
                    "command -v flower-supernode >/dev/null"
                ),
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
                timeout_sec=20.0,
                check=True,
            )
            remote_archive = f"{remote_runtime_dir}/comcast-anomaly-app.tar.gz"
            _copy_to_remote(vm, archive_path, remote_archive, azure.ssh_connect_timeout_sec)
            _run_remote_command(
                vm=vm,
                remote_cmd=(
                    f"rm -rf -- {shlex.quote(remote_app_dir)} && "
                    f"mkdir -p {shlex.quote(remote_app_dir)} && "
                    f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_app_dir)}"
                ),
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
                timeout_sec=60.0,
                check=True,
            )
            _run_remote_command(
                vm=vm,
                remote_cmd=(
                    f"{shlex.quote(azure.remote_python)} -m pip install -e {shlex.quote(remote_app_dir)} "
                    "--quiet"
                ),
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
                timeout_sec=300.0,
                check=True,
            )

        # Copy TLS cert material.
        ca_remote_path = f"{remote_runtime_dir}/{tls.remote_cert_dir.rstrip('/')}/ca.crt"
        server_cert_remote_path = f"{remote_runtime_dir}/{tls.remote_cert_dir.rstrip('/')}/server.crt"
        server_key_remote_path = f"{remote_runtime_dir}/{tls.remote_cert_dir.rstrip('/')}/server.key"
        for vm in azure.vms:
            _copy_to_remote(
                vm=vm,
                local_path=Path(tls.ca_cert_local_path),
                remote_path=ca_remote_path,
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
            )
        _copy_to_remote(
            vm=superlink_vm,
            local_path=Path(tls.server_cert_local_path),
            remote_path=server_cert_remote_path,
            connect_timeout_sec=azure.ssh_connect_timeout_sec,
        )
        _copy_to_remote(
            vm=superlink_vm,
            local_path=Path(tls.server_key_local_path),
            remote_path=server_key_remote_path,
            connect_timeout_sec=azure.ssh_connect_timeout_sec,
        )

        # Partition mapping and key distribution.
        partition_plan = _build_partition_plan(cfg)
        for entry in partition_plan:
            partition_id = int(entry["partition_id"])
            vm = vm_by_name[str(entry["vm_name"])]
            remote_priv = f"{remote_runtime_dir}/{auth.remote_key_dir.rstrip('/')}/sn_{partition_id:04d}_private.pem"
            remote_pub = f"{remote_runtime_dir}/{auth.remote_key_dir.rstrip('/')}/sn_{partition_id:04d}_public.pem"
            entry["private_key_remote_path"] = remote_priv
            entry["public_key_remote_path"] = remote_pub

            _copy_to_remote(
                vm=vm,
                local_path=Path(str(entry["private_key_local_path"])),
                remote_path=remote_priv,
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
            )
            _copy_to_remote(
                vm=control_vm,
                local_path=Path(str(entry["public_key_local_path"])),
                remote_path=remote_pub,
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
            )

        control_port = _reserve_remote_port(superlink_vm, azure.remote_python, azure.ssh_connect_timeout_sec)
        fleet_port = _reserve_remote_port(superlink_vm, azure.remote_python, azure.ssh_connect_timeout_sec)
        serverappio_port = _reserve_remote_port(superlink_vm, azure.remote_python, azure.ssh_connect_timeout_sec)

        superlink_bind_host = azure.superlink_bind_host or superlink_vm.host
        superlink_cmd = _make_superlink_cmd(
            control_api_addr=f"{superlink_bind_host}:{control_port}",
            fleet_api_addr=f"{superlink_bind_host}:{fleet_port}",
            serverappio_api_addr=f"{superlink_bind_host}:{serverappio_port}",
            ca_cert=ca_remote_path,
            server_cert=server_cert_remote_path,
            server_key=server_key_remote_path,
            storage_dir=f"{remote_runtime_dir}/storage",
            database=str(cfg.deployment.local_database),
        )
        superlink_pid_file = f"{remote_runtime_dir}/pids/superlink.pid"
        _start_remote_process(
            vm=superlink_vm,
            cmd=superlink_cmd,
            pid_file=superlink_pid_file,
            log_file=f"{remote_runtime_dir}/logs/superlink.log",
            connect_timeout_sec=azure.ssh_connect_timeout_sec,
        )

        control_api_addr = f"{superlink_vm.host}:{control_port}"
        fleet_api_addr = f"{superlink_vm.host}:{fleet_port}"
        serverappio_api_addr = f"{superlink_vm.host}:{serverappio_port}"
        _wait_for_remote_tcp(
            vm=control_vm,
            target_host=superlink_vm.host,
            target_port=control_port,
            remote_python=azure.remote_python,
            timeout_sec=azure.startup_timeout_sec,
            connect_timeout_sec=azure.ssh_connect_timeout_sec,
        )
        _wait_for_remote_tcp(
            vm=control_vm,
            target_host=superlink_vm.host,
            target_port=fleet_port,
            remote_python=azure.remote_python,
            timeout_sec=azure.startup_timeout_sec,
            connect_timeout_sec=azure.ssh_connect_timeout_sec,
        )

        # Configure connection on control VM.
        control_flwr_home = f"{remote_runtime_dir}/flwr_home"
        handle = ManagedAzureRuntimeHandle(
            connection_name=cfg.deployment.connection_name,
            run_name=cfg.artifacts.run_name,
            local_runtime_dir=local_runtime_dir,
            local_logs_dir=local_logs_dir,
            runtime_state_path=runtime_state_path,
            local_artifacts_root=local_artifacts_root,
            remote_workspace_dir=azure.remote_workspace_dir,
            remote_app_dir=remote_app_dir,
            remote_runtime_dir=remote_runtime_dir,
            remote_artifacts_root=remote_artifacts_root,
            control_vm=control_vm,
            superlink_vm=superlink_vm,
            vm_by_name=vm_by_name,
            control_api_addr=control_api_addr,
            fleet_api_addr=fleet_api_addr,
            serverappio_api_addr=serverappio_api_addr,
            control_flwr_home=control_flwr_home,
            tls_remote_paths={
                "ca_cert": ca_remote_path,
                "server_cert": server_cert_remote_path,
                "server_key": server_key_remote_path,
            },
            superlink_pid_file=superlink_pid_file,
            supernode_entries=partition_plan,
            startup_timeout_sec=int(azure.startup_timeout_sec),
            poll_interval_sec=float(azure.poll_interval_sec),
            domain_run_timeout_sec=int(azure.domain_run_timeout_sec),
            teardown_grace_sec=int(azure.teardown_grace_sec),
            ssh_connect_timeout_sec=int(azure.ssh_connect_timeout_sec),
        )
        config_toml = "\n".join(
            [
                "[superlink]",
                f'default = "{cfg.deployment.connection_name}"',
                "",
                f"[superlink.{cfg.deployment.connection_name}]",
                f'address = "{control_api_addr}"',
                f'root-certificates = "{ca_remote_path}"',
                "insecure = false",
                "",
            ]
        )
        write_cfg_cmd = (
            f"mkdir -p {shlex.quote(control_flwr_home)} && "
            f"cat > {shlex.quote(control_flwr_home + '/config.toml')} <<'EOF'\n"
            f"{config_toml}"
            "EOF\n"
        )
        _run_remote_command(
            vm=control_vm,
            remote_cmd=write_cfg_cmd,
            connect_timeout_sec=azure.ssh_connect_timeout_sec,
            timeout_sec=20.0,
            check=True,
        )
        _write_local_runtime_state(handle, extra={"started_at": _utc_now()})

        # Register all public keys.
        for entry in handle.supernode_entries:
            proc = _run_control_flwr(
                handle=handle,
                args=[
                    "flwr",
                    "supernode",
                    "register",
                    str(entry["public_key_remote_path"]),
                    handle.connection_name,
                    "--format",
                    "json",
                ],
                timeout_sec=60.0,
                check=True,
            )
            try:
                payload = _parse_json_payload(proc.stdout)
            except json.JSONDecodeError:
                payload = {"success": False, "raw": proc.stdout}
            if not payload.get("success", False):
                raise RuntimeError(f"SuperNode registration failed for partition {entry['partition_id']}: {proc.stdout}")

        # Launch supernodes.
        for entry in handle.supernode_entries:
            vm = handle.vm_by_name[str(entry["vm_name"])]
            clientappio_port = _reserve_remote_port(vm, azure.remote_python, azure.ssh_connect_timeout_sec)
            entry["clientappio_port"] = int(clientappio_port)
            clientappio_addr = f"127.0.0.1:{clientappio_port}"
            entry["clientappio_addr"] = f"{vm.host}:{clientappio_port}"
            pid_file = f"{remote_runtime_dir}/pids/supernode_{int(entry['partition_id']):04d}.pid"
            log_file = f"{remote_runtime_dir}/logs/supernode_{int(entry['partition_id']):04d}.log"
            entry["pid_file"] = pid_file
            entry["log_file"] = log_file
            cmd = _make_supernode_cmd(
                fleet_api_addr=fleet_api_addr,
                root_cert=ca_remote_path,
                private_key=str(entry["private_key_remote_path"]),
                clientappio_api_addr=clientappio_addr,
                partition_id=int(entry["partition_id"]),
                num_partitions=int(azure.total_supernodes),
            )
            _start_remote_process(
                vm=vm,
                cmd=cmd,
                pid_file=pid_file,
                log_file=log_file,
                connect_timeout_sec=azure.ssh_connect_timeout_sec,
            )
            started_supernode_pid_files.append((vm, pid_file))

        _wait_for_supernodes_online(handle=handle, expected_online=int(azure.total_supernodes))
        state = _write_local_runtime_state(
            handle,
            extra={
                "startup": {
                    "online_supernodes_required": int(azure.total_supernodes),
                    "status": "ready",
                }
            },
        )
        emit_hook(
            hook_sink,
            event_type="runtime.started",
            payload=state,
            run_name=cfg.artifacts.run_name,
            domain=None,
        )
        emit_hook(
            hook_sink,
            event_type="supernodes.updated",
            payload={
                "connection_name": handle.connection_name,
                "nodes": [
                    {
                        "node_id": str(e["partition_id"]),
                        "status": "online",
                        "owner_name": e["vm_name"],
                    }
                    for e in handle.supernode_entries
                ],
            },
            run_name=cfg.artifacts.run_name,
            domain=None,
        )
        return handle
    except Exception:
        if handle is not None:
            try:
                stop_managed_azure_runtime(handle, quiet=True, raise_on_incomplete=False)
            except Exception:
                pass
        else:
            _best_effort_cleanup_on_start_failure(
                superlink_vm=superlink_vm,
                superlink_pid_file=superlink_pid_file,
                started_supernode_pid_files=started_supernode_pid_files,
                grace_sec=int(azure.teardown_grace_sec),
                connect_timeout_sec=int(azure.ssh_connect_timeout_sec),
            )
        raise
    finally:
        if archive_path.exists():
            archive_path.unlink()
            shutil.rmtree(archive_path.parent, ignore_errors=True)


def submit_run_and_wait_remote(
    handle: ManagedAzureRuntimeHandle,
    domain: str,
    run_config_toml_text: str,
    cfg: ExperimentConfig,
    hook_sink: UiHookSink | None = None,
) -> dict[str, Any]:
    """Submit one Flower run from control VM and wait for terminal status."""
    remote_cfg_path = f"{handle.remote_runtime_dir}/run_configs/{domain}_run_config.toml"
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as tmp:
        tmp.write(run_config_toml_text)
        local_tmp_cfg = Path(tmp.name)
    try:
        _copy_to_remote(
            vm=handle.control_vm,
            local_path=local_tmp_cfg,
            remote_path=remote_cfg_path,
            connect_timeout_sec=handle.ssh_connect_timeout_sec,
        )
    finally:
        if local_tmp_cfg.exists():
            local_tmp_cfg.unlink()

    run_cmd = [
        "flwr",
        "run",
        handle.remote_app_dir,
        handle.connection_name,
        "--run-config",
        remote_cfg_path,
        "--format",
        "json",
    ]
    if cfg.deployment.federation:
        run_cmd.extend(["--federation", str(cfg.deployment.federation)])

    proc = _run_control_flwr(
        handle=handle,
        args=run_cmd,
        timeout_sec=float(handle.domain_run_timeout_sec),
        check=True,
    )
    run_id = _extract_run_id(proc.stdout)
    run_payload = _parse_json_payload(proc.stdout)
    emit_hook(
        hook_sink,
        event_type="run.started",
        payload={"run_id": run_id, "connection_name": handle.connection_name},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )

    deadline = time.monotonic() + float(handle.domain_run_timeout_sec)
    last_status = "unknown"
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        ls_proc = _run_control_flwr(
            handle=handle,
            args=["flwr", "ls", handle.connection_name, "--runs", "--format", "json"],
            timeout_sec=30.0,
            check=True,
        )
        payload = _parse_json_payload(ls_proc.stdout)
        last_payload = payload
        runs = payload.get("runs", [])
        found = next((r for r in runs if int(r.get("run-id", -1)) == run_id), None)
        if found is not None:
            last_status = str(found.get("status", "unknown"))
            emit_hook(
                hook_sink,
                event_type="run.status",
                payload={"run_id": run_id, "status": last_status},
                run_name=cfg.artifacts.run_name,
                domain=domain,
            )
            if last_status.startswith("finished:"):
                if last_status == "finished:completed":
                    emit_hook(
                        hook_sink,
                        event_type="run.completed",
                        payload={"run_id": run_id, "status": last_status},
                        run_name=cfg.artifacts.run_name,
                        domain=domain,
                    )
                    return {
                        "run_id": run_id,
                        "status": last_status,
                        "run_payload": run_payload,
                        "status_payload": payload,
                    }
                emit_hook(
                    hook_sink,
                    event_type="run.failed",
                    payload={"run_id": run_id, "status": last_status},
                    run_name=cfg.artifacts.run_name,
                    domain=domain,
                )
                raise RuntimeError(
                    "Remote run finished in non-success state.\n"
                    f"run_id={run_id}, status={last_status}\n"
                    f"status_payload={json.dumps(payload)}"
                )
        time.sleep(float(handle.poll_interval_sec))

    emit_hook(
        hook_sink,
        event_type="run.timeout",
        payload={"run_id": run_id, "status": last_status, "timeout_sec": handle.domain_run_timeout_sec},
        run_name=cfg.artifacts.run_name,
        domain=domain,
    )
    raise TimeoutError(
        "Timed out waiting for remote deployment run completion.\n"
        f"run_id={run_id}, last_status={last_status}\n"
        f"last_payload={json.dumps(last_payload)}"
    )


def collect_remote_artifacts(handle: ManagedAzureRuntimeHandle, cfg: ExperimentConfig) -> None:
    """Mirror remote run artifacts to the local artifact root."""
    local_run_root = Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name
    ensure_dir(local_run_root)
    _copy_from_remote(
        vm=handle.superlink_vm,
        remote_path=f"{handle.remote_artifacts_root}/{cfg.artifacts.run_name}/.",
        local_path=local_run_root,
        connect_timeout_sec=handle.ssh_connect_timeout_sec,
    )


def stop_managed_azure_runtime(
    handle: ManagedAzureRuntimeHandle,
    quiet: bool = False,
    raise_on_incomplete: bool = True,
) -> None:
    """Stop all managed Azure runtime processes and sync logs/state."""
    supernode_results: list[dict[str, Any]] = []
    for entry in reversed(handle.supernode_entries):
        vm = handle.vm_by_name[str(entry["vm_name"])]
        pid_file = str(entry.get("pid_file", ""))
        if not pid_file:
            continue
        out = _stop_remote_process(
            vm=vm,
            pid_file=pid_file,
            grace_sec=int(handle.teardown_grace_sec),
            connect_timeout_sec=handle.ssh_connect_timeout_sec,
        )
        out["partition_id"] = int(entry["partition_id"])
        supernode_results.append(out)

    superlink_result = _stop_remote_process(
        vm=handle.superlink_vm,
        pid_file=handle.superlink_pid_file,
        grace_sec=int(handle.teardown_grace_sec),
        connect_timeout_sec=handle.ssh_connect_timeout_sec,
    )

    # Copy remote logs to local runtime log mirror.
    for vm in handle.vm_by_name.values():
        local_vm_logs = handle.local_logs_dir / vm.name
        ensure_dir(local_vm_logs)
        try:
            _copy_from_remote(
                vm=vm,
                remote_path=f"{handle.remote_runtime_dir}/logs/",
                local_path=local_vm_logs,
                connect_timeout_sec=handle.ssh_connect_timeout_sec,
            )
        except Exception:
            # best-effort log sync
            pass

    all_exited = (
        superlink_result.get("status") in {"EXITED", "MISSING_PID"}
        and all(x.get("status") in {"EXITED", "MISSING_PID"} for x in supernode_results)
    )
    state = _write_local_runtime_state(
        handle,
        extra={
            "stopped_at": _utc_now(),
            "teardown": {
                "superlink": superlink_result,
                "supernodes": list(reversed(supernode_results)),
                "all_exited": bool(all_exited),
            },
        },
    )
    if not quiet:
        print("Managed Azure runtime teardown:", json.dumps(state["teardown"], sort_keys=True))
    if not all_exited and raise_on_incomplete:
        raise RuntimeError(
            "Managed Azure runtime teardown incomplete; remote processes may still be running.\n"
            f"runtime_state={handle.runtime_state_path}"
        )


def make_remote_run_config_toml(cfg: ExperimentConfig, domain: str, remote_output_root: str) -> str:
    """Build run-config TOML for remote execution with remote output root."""
    cfg_copy = loads_config_json(json.dumps(cfg.to_dict(), separators=(",", ":")))
    cfg_copy.artifacts.root_dir = remote_output_root
    from .app_state import build_run_config_payload  # local import to avoid cycle

    payload = build_run_config_payload(cfg_copy, domain)
    lines = [f"{k} = {json.dumps(v)}" for k, v in payload.items()]
    return "\n".join(lines) + "\n"
