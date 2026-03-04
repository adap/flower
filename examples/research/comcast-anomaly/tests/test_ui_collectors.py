from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from comcast_ui.collectors.file_poller import FilePoller
from comcast_ui.collectors.supernode_poller import SupernodePoller
from comcast_ui.schemas import make_event
from comcast_ui.state import UiStateStore


class _CaptureSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event_type, payload, run_name=None, domain=None) -> None:  # type: ignore[no-untyped-def]
        self.events.append(
            {
                "event_type": event_type,
                "payload": payload,
                "run_name": run_name,
                "domain": domain,
            }
        )


def test_file_poller_emits_runtime_metrics_and_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "fl"
    run_name = "ui_test"
    base = run_root / run_name
    (base / "deployment_runtime").mkdir(parents=True)
    (base / "downstream_rxmer").mkdir(parents=True)

    runtime_state = {
        "num_supernodes": 2,
        "processes": {"supernode_pids": [10, 11]},
        "started_at": "2026-03-02T00:00:00+00:00",
        "updated_at": "2026-03-02T00:00:00+00:00",
    }
    metrics = {
        "domain": "downstream_rxmer",
        "unknown_threshold": 0.3,
        "raw_metrics": {"macro_f1": 0.4, "event_peak_macro_f1": 0.3, "anomaly_auroc": 0.6},
        "gated_metrics": {"macro_f1": 0.5, "event_peak_macro_f1": 0.4, "anomaly_auroc": 0.7},
    }
    summary = {"domains": {"downstream_rxmer": metrics}}

    (base / "deployment_runtime" / "runtime_state.json").write_text(
        json.dumps(runtime_state), encoding="utf-8"
    )
    (base / "downstream_rxmer" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    sink = _CaptureSink()
    poller = FilePoller(run_root=run_root, run_name=run_name, domains=["downstream_rxmer"], sink=sink, interval_sec=0.01)
    poller._scan_once()  # pylint: disable=protected-access

    event_types = [e["event_type"] for e in sink.events]
    assert "artifact.discovered" in event_types
    assert "runtime.started" in event_types
    assert "metrics.updated" in event_types
    assert "domain.completed" in event_types
    assert "run.completed" in event_types


def test_panel_payloads_implemented_and_stub() -> None:
    store = UiStateStore()
    store.apply_event(
        make_event(
            event_type="metrics.updated",
            payload={
                "metrics": {
                    "raw_metrics": {"macro_f1": 0.1, "event_peak_macro_f1": 0.2, "anomaly_auroc": 0.3},
                    "gated_metrics": {"macro_f1": 0.4, "event_peak_macro_f1": 0.5, "anomaly_auroc": 0.6},
                    "unknown_threshold": 0.2,
                },
                "domain": "downstream_rxmer",
            },
            run_name="x",
            domain="downstream_rxmer",
        )
    )

    implemented = store.get_panel_snapshot("global_quality_trends")
    stub = store.get_panel_snapshot("unknown_gate_monitor")
    assert implemented["panel_id"] == "global_quality_trends"
    assert "domains" in implemented
    assert stub["status"] == "stub"


def test_topology_panel_uses_supernodes_not_domains() -> None:
    store = UiStateStore()
    store.apply_event(
        make_event(
            event_type="domain.completed",
            payload={"status": "finished:completed"},
            run_name="x",
            domain="downstream_rxmer",
        )
    )
    store.apply_event(
        make_event(
            event_type="supernodes.updated",
            payload={
                "connection_name": "comcast-local",
                "nodes": [
                    {"node_id": "0", "status": "online"},
                    {"node_id": "1", "status": "offline"},
                ],
            },
            run_name="x",
            domain=None,
        )
    )

    topology = store.get_panel_snapshot("federation_topology")
    node_ids = [n["id"] for n in topology["nodes"]]
    assert "superlink" in node_ids
    assert "supernode-0" in node_ids
    assert "supernode-1" in node_ids
    assert all(not nid.startswith("domain-") for nid in node_ids)


def test_supernode_poller_emits_updates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "fl"
    run_name = "ui_test"
    runtime_dir = run_root / run_name / "deployment_runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "runtime_state.json").write_text(
        json.dumps({"connection_name": "comcast-local", "flwr_home": "/tmp/flwr-home"}),
        encoding="utf-8",
    )

    sink = _CaptureSink()
    poller = SupernodePoller(run_root=run_root, run_name=run_name, sink=sink, interval_sec=0.01)

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return subprocess.CompletedProcess(
            ["flwr"],
            0,
            stdout=json.dumps(
                {
                    "success": True,
                    "nodes": [
                        {"node-id": "0", "status": "online", "owner-name": "test"},
                        {"node-id": "1", "status": "offline", "owner-name": "test"},
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("comcast_ui.collectors.supernode_poller.subprocess.run", _fake_run)
    poller._poll_once()  # pylint: disable=protected-access

    events = [e for e in sink.events if e["event_type"] == "supernodes.updated"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["connection_name"] == "comcast-local"
    assert len(payload["nodes"]) == 2
