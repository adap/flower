from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from comcast_fl.adapters import submit_run_and_wait

try:
    import fastapi  # noqa: F401

    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


class _CaptureSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, str | None, dict]] = []

    def emit(self, event_type, payload, run_name=None, domain=None) -> None:  # type: ignore[no-untyped-def]
        self.events.append((event_type, run_name, domain, payload))


def test_submit_run_and_wait_emits_hook_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"ls": 0}

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
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

    sink = _CaptureSink()
    out = submit_run_and_wait(
        app_dir=tmp_path,
        connection_name="local-test",
        run_config_toml=run_cfg,
        timeout_sec=10,
        poll_sec=0.01,
        env={},
        run_name="demo",
        domain="downstream_rxmer",
        hook_sink=sink,
    )
    assert out["run_id"] == 42

    types = [x[0] for x in sink.events]
    assert "run.started" in types
    assert "run.status" in types
    assert "run.completed" in types


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
def test_ui_api_and_ws_contract(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from comcast_ui.app import create_app

    run_root = tmp_path / "artifacts" / "fl"
    app = create_app(run_root=run_root, run_name="ui_api", domains=["downstream_rxmer"], poll_interval_sec=0.01)

    with TestClient(app) as client:
        layout = client.get("/api/v1/layout")
        assert layout.status_code == 200
        body = layout.json()
        assert len(body["panels"]) == 10

        state = client.get("/api/v1/state")
        assert state.status_code == 200

        panel = client.get("/api/v1/panels/federation_topology")
        assert panel.status_code == 200
        assert panel.json()["panel_id"] == "federation_topology"

        with client.websocket_connect("/api/v1/events") as ws:
            app.state.ui_sink.emit(
                "domain.started",
                {"mode": "simulation"},
                run_name="ui_api",
                domain="downstream_rxmer",
            )
            msg = ws.receive_text()
            evt = json.loads(msg)
            assert evt["event_type"] in {"domain.started", "artifact.discovered"}
