from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from comcast_fl.adapters import submit_run_and_wait
from comcast_fl.config import AzureSshConfig, AzureVmSpec, SupernodeAuthConfig, load_experiment_config
from comcast_fl.deployment_azure_ssh import (
    ManagedAzureRuntimeHandle,
    _build_partition_plan,
    _make_superlink_cmd,
    _make_supernode_cmd,
    stop_managed_azure_runtime,
    submit_run_and_wait_remote,
)
from comcast_fl.deployment_local import start_managed_local_runtime, stop_managed_local_runtime


BASE = Path(__file__).resolve().parents[1]


class _FakeProc:
    _pid_seq = 20000

    def __init__(self) -> None:
        _FakeProc._pid_seq += 1
        self.pid = _FakeProc._pid_seq
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def kill(self) -> None:
        self._returncode = -9


def test_managed_local_runtime_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg.mode = "deployment"
    cfg.deployment.launch_mode = "managed_local"
    cfg.federation.num_clients = 3
    cfg.deployment.local_num_supernodes = None
    cfg.artifacts.root_dir = str(tmp_path)
    cfg.artifacts.run_name = "runtime-unit"

    cmds: list[list[str]] = []

    def _fake_start(
        cmd: list[str],
        log_path: Path,
        env: dict[str, str],
        cwd: Path | None = None,
    ) -> _FakeProc:
        del log_path, env, cwd
        cmds.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(
        "comcast_fl.deployment_local._start_logged_process",
        _fake_start,
    )
    ports = iter(range(41000, 41000 + 16))
    monkeypatch.setattr("comcast_fl.deployment_local._reserve_free_port", lambda: next(ports))
    monkeypatch.setattr("comcast_fl.deployment_local._wait_for_tcp", lambda *args, **kwargs: None)
    monkeypatch.setattr("comcast_fl.deployment_local.time.sleep", lambda _: None)

    handle = start_managed_local_runtime(cfg)
    assert handle.num_supernodes == 3
    assert len(handle.supernode_procs) == 3
    assert cmds[0][0] == "flower-superlink"

    for idx in range(3):
        node_cfg = cmds[idx + 1][cmds[idx + 1].index("--node-config") + 1]
        assert f"partition-id={idx}" in node_cfg
        assert "num-partitions=3" in node_cfg
        assert f"client-id={idx}" in node_cfg

    flwr_cfg_path = handle.flwr_home / "config.toml"
    text = flwr_cfg_path.read_text(encoding="utf-8")
    assert f'default = "{cfg.deployment.connection_name}"' in text
    assert f'address = "{handle.control_api_addr}"' in text

    stop_managed_local_runtime(handle, shutdown_grace_sec=1)
    state = (handle.runtime_dir / "runtime_state.json").read_text(encoding="utf-8")
    assert '"all_exited": true' in state


def test_submit_run_and_wait_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"ls": 0}

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        cmd = args[0]
        if cmd[:2] == ["flwr", "run"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"success": true, "run-id": 42}',
                stderr="",
            )
        if cmd[:2] == ["flwr", "ls"]:
            calls["ls"] += 1
            status = "running" if calls["ls"] == 1 else "finished:completed"
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f'{{"runs":[{{"run-id":42,"status":"{status}"}}]}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("comcast_fl.adapters.subprocess.run", _fake_run)
    monkeypatch.setattr("comcast_fl.adapters.time.sleep", lambda _: None)

    run_cfg = tmp_path / "run-config.toml"
    run_cfg.write_text('domain = "downstream_rxmer"\n', encoding="utf-8")
    out = submit_run_and_wait(
        app_dir=tmp_path,
        connection_name="local-test",
        run_config_toml=run_cfg,
        timeout_sec=10,
        poll_sec=0.01,
        env={},
    )
    assert out["run_id"] == 42
    assert out["status"] == "finished:completed"


def test_submit_run_and_wait_failure_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        cmd = args[0]
        if cmd[:2] == ["flwr", "run"]:
            return subprocess.CompletedProcess(cmd, 0, stdout='{"run-id": 3}', stderr="")
        if cmd[:2] == ["flwr", "ls"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"runs":[{"run-id":3,"status":"finished:failed"}]}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("comcast_fl.adapters.subprocess.run", _fake_run)
    monkeypatch.setattr("comcast_fl.adapters.time.sleep", lambda _: None)

    run_cfg = tmp_path / "run-config.toml"
    run_cfg.write_text('domain = "downstream_rxmer"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-success"):
        submit_run_and_wait(
            app_dir=tmp_path,
            connection_name="local-test",
            run_config_toml=run_cfg,
            timeout_sec=10,
            poll_sec=0.01,
            env={},
        )


def test_submit_run_and_wait_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ticks = {"v": 0}

    def _mono() -> float:
        ticks["v"] += 1
        return float(ticks["v"])

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        cmd = args[0]
        if cmd[:2] == ["flwr", "run"]:
            return subprocess.CompletedProcess(cmd, 0, stdout='{"run-id": 3}', stderr="")
        if cmd[:2] == ["flwr", "ls"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"runs":[{"run-id":3,"status":"running"}]}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("comcast_fl.adapters.subprocess.run", _fake_run)
    monkeypatch.setattr("comcast_fl.adapters.time.sleep", lambda _: None)
    monkeypatch.setattr("comcast_fl.adapters.time.monotonic", _mono)

    run_cfg = tmp_path / "run-config.toml"
    run_cfg.write_text('domain = "downstream_rxmer"\n', encoding="utf-8")
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        submit_run_and_wait(
            app_dir=tmp_path,
            connection_name="local-test",
            run_config_toml=run_cfg,
            timeout_sec=2,
            poll_sec=0.01,
            env={},
        )


def test_azure_command_builders_include_tls_and_auth() -> None:
    s_cmd = _make_superlink_cmd(
        control_api_addr="0.0.0.0:39093",
        fleet_api_addr="0.0.0.0:39094",
        serverappio_api_addr="0.0.0.0:39095",
        ca_cert="/remote/ca.crt",
        server_cert="/remote/server.pem",
        server_key="/remote/server.key",
        storage_dir="/remote/storage",
        database=":flwr-in-memory:",
    )
    joined = " ".join(s_cmd)
    assert "--ssl-ca-certfile /remote/ca.crt" in joined
    assert "--ssl-certfile /remote/server.pem" in joined
    assert "--ssl-keyfile /remote/server.key" in joined
    assert "--enable-supernode-auth" in joined

    n_cmd = _make_supernode_cmd(
        fleet_api_addr="10.0.0.4:39094",
        root_cert="/remote/ca.crt",
        private_key="/remote/sn0.pem",
        clientappio_api_addr="0.0.0.0:41000",
        partition_id=0,
        num_partitions=4,
    )
    n_joined = " ".join(n_cmd)
    assert "--root-certificates /remote/ca.crt" in n_joined
    assert "--auth-supernode-private-key /remote/sn0.pem" in n_joined
    assert "partition-id=0 num-partitions=4 client-id=0" in n_joined


def test_partition_plan_deterministic() -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg.mode = "deployment"
    cfg.federation.num_clients = 3
    cfg.deployment.launch_mode = "managed_azure_ssh"
    cfg.deployment.azure_ssh = AzureSshConfig(
        vms=[
            AzureVmSpec(name="a", host="10.0.0.1", ssh_user="u", ssh_key_path="/tmp/k", supernodes_on_vm=2),
            AzureVmSpec(name="b", host="10.0.0.2", ssh_user="u", ssh_key_path="/tmp/k", supernodes_on_vm=1),
        ],
        total_supernodes=3,
        control_vm="a",
        superlink_vm="a",
    )
    cfg.deployment.supernode_auth = SupernodeAuthConfig(
        enabled=True,
        private_key_local_paths=["p0", "p1", "p2"],
        public_key_local_paths=["u0", "u1", "u2"],
    )
    plan = _build_partition_plan(cfg)
    assert [p["partition_id"] for p in plan] == [0, 1, 2]
    assert [p["vm_name"] for p in plan] == ["a", "a", "b"]


def _fake_azure_handle(tmp_path: Path) -> ManagedAzureRuntimeHandle:
    vm = AzureVmSpec(
        name="vm-a",
        host="10.0.0.4",
        ssh_port=22,
        ssh_user="azureuser",
        ssh_key_path="/tmp/id",
        supernodes_on_vm=1,
        roles=["superlink", "control", "worker"],
    )
    return ManagedAzureRuntimeHandle(
        connection_name="comcast-azure",
        run_name="test-run",
        local_runtime_dir=tmp_path / "runtime",
        local_logs_dir=tmp_path / "runtime" / "logs",
        runtime_state_path=tmp_path / "runtime" / "runtime_state.json",
        local_artifacts_root=tmp_path / "artifacts",
        remote_workspace_dir="/opt/comcast",
        remote_app_dir="/opt/comcast/app",
        remote_runtime_dir="/opt/comcast/runtime/test-run",
        remote_artifacts_root="/opt/comcast/runtime/test-run/artifacts/fl",
        control_vm=vm,
        superlink_vm=vm,
        vm_by_name={"vm-a": vm},
        control_api_addr="10.0.0.4:39093",
        fleet_api_addr="10.0.0.4:39094",
        serverappio_api_addr="10.0.0.4:39095",
        control_flwr_home="/opt/comcast/runtime/test-run/flwr_home",
        tls_remote_paths={"ca_cert": "/opt/comcast/runtime/test-run/secrets/certificates/ca.crt"},
        superlink_pid_file="/opt/comcast/runtime/test-run/pids/superlink.pid",
        supernode_entries=[],
        startup_timeout_sec=10,
        poll_interval_sec=0.01,
        domain_run_timeout_sec=3,
        teardown_grace_sec=1,
        ssh_connect_timeout_sec=1,
    )


def test_submit_run_and_wait_remote_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    handle = _fake_azure_handle(tmp_path)
    calls = {"ls": 0}

    def _fake_run_control_flwr(handle, args, timeout_sec=None, check=True):  # type: ignore[no-untyped-def]
        del handle, timeout_sec, check
        if args[:2] == ["flwr", "run"]:
            return subprocess.CompletedProcess(args, 0, stdout='{"success": true, "run-id": 99}', stderr="")
        if args[:2] == ["flwr", "ls"]:
            calls["ls"] += 1
            status = "running" if calls["ls"] == 1 else "finished:completed"
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=f'{{"runs":[{{"run-id":99,"status":"{status}"}}]}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected args: {args}")

    monkeypatch.setattr("comcast_fl.deployment_azure_ssh._copy_to_remote", lambda *a, **k: None)
    monkeypatch.setattr("comcast_fl.deployment_azure_ssh._run_control_flwr", _fake_run_control_flwr)
    monkeypatch.setattr("comcast_fl.deployment_azure_ssh.time.sleep", lambda _: None)
    out = submit_run_and_wait_remote(
        handle=handle,
        domain="downstream_rxmer",
        run_config_toml_text='domain = "downstream_rxmer"\n',
        cfg=cfg,
    )
    assert out["run_id"] == 99
    assert out["status"] == "finished:completed"


def test_submit_run_and_wait_remote_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    handle = _fake_azure_handle(tmp_path)

    def _fake_run_control_flwr(handle, args, timeout_sec=None, check=True):  # type: ignore[no-untyped-def]
        del handle, timeout_sec, check
        if args[:2] == ["flwr", "run"]:
            return subprocess.CompletedProcess(args, 0, stdout='{"run-id": 7}', stderr="")
        if args[:2] == ["flwr", "ls"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout='{"runs":[{"run-id":7,"status":"finished:failed"}]}',
                stderr="",
            )
        raise AssertionError(f"Unexpected args: {args}")

    monkeypatch.setattr("comcast_fl.deployment_azure_ssh._copy_to_remote", lambda *a, **k: None)
    monkeypatch.setattr("comcast_fl.deployment_azure_ssh._run_control_flwr", _fake_run_control_flwr)
    monkeypatch.setattr("comcast_fl.deployment_azure_ssh.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="non-success"):
        submit_run_and_wait_remote(
            handle=handle,
            domain="downstream_rxmer",
            run_config_toml_text='domain = "downstream_rxmer"\n',
            cfg=cfg,
        )


def test_stop_managed_azure_runtime_raises_when_not_all_exited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handle = _fake_azure_handle(tmp_path)
    handle.supernode_entries = [{"partition_id": 0, "vm_name": "vm-a", "pid_file": "/tmp/sn.pid"}]
    (handle.local_logs_dir / "vm-a").mkdir(parents=True, exist_ok=True)

    calls = {"n": 0}

    def _fake_stop(vm, pid_file, grace_sec, connect_timeout_sec):  # type: ignore[no-untyped-def]
        del vm, pid_file, grace_sec, connect_timeout_sec
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "RUNNING"}
        return {"status": "EXITED"}

    monkeypatch.setattr("comcast_fl.deployment_azure_ssh._stop_remote_process", _fake_stop)
    monkeypatch.setattr("comcast_fl.deployment_azure_ssh._copy_from_remote", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="teardown incomplete"):
        stop_managed_azure_runtime(handle, quiet=True, raise_on_incomplete=True)
