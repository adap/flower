"""In-memory UI state with event ring buffers and panel snapshots."""

from __future__ import annotations

from collections import deque
import copy
import threading
from typing import Any

from .panels.quality import build_quality_payload
from .panels.registry import PANEL_IDS, get_panel_spec
from .panels.stubs import stub_panel_payload
from .panels.timeline import build_timeline_payload
from .panels.topology import build_topology_payload
from .schemas import UiEventV1


class UiStateStore:
    """Thread-safe state store updated from hooks and file polling."""

    def __init__(self, max_events: int = 1500) -> None:
        self._lock = threading.RLock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._timeline: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._run_name: str | None = None
        self._runtime: dict[str, Any] = {
            "state": "idle",
            "supernodes": [],
        }
        self._domains: dict[str, dict[str, Any]] = {}
        self._quality: dict[str, list[dict[str, Any]]] = {}
        self._quality_index: dict[str, int] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._run_status: dict[str, Any] = {"status": "idle"}

    def apply_event(self, event: UiEventV1) -> None:
        with self._lock:
            event_dict = self._event_dict(event)
            self._events.append(event_dict)
            self._timeline.append(event_dict)
            if event.run_name:
                self._run_name = event.run_name
            self._apply_event(event)

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)[-limit:]

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            base = self._snapshot_base()
        panels = {panel_id: self.get_panel_snapshot(panel_id) for panel_id in PANEL_IDS}
        base["panels"] = panels
        return base

    def get_panel_snapshot(self, panel_id: str) -> dict[str, Any]:
        spec = get_panel_spec(panel_id)
        with self._lock:
            state = self._snapshot_base()
        if spec.id == "federation_topology":
            return build_topology_payload(state)
        if spec.id == "round_timeline":
            return build_timeline_payload(state)
        if spec.id == "global_quality_trends":
            return build_quality_payload(state)
        return stub_panel_payload(panel_id)

    def _snapshot_base(self) -> dict[str, Any]:
        return {
            "run_name": self._run_name,
            "run_status": copy.deepcopy(self._run_status),
            "runtime": copy.deepcopy(self._runtime),
            "domains": copy.deepcopy(self._domains),
            "quality": copy.deepcopy(self._quality),
            "artifacts": copy.deepcopy(self._artifacts),
            "timeline": list(self._timeline),
            "recent_events": list(self._events)[-200:],
        }

    def _apply_event(self, event: UiEventV1) -> None:
        et = event.event_type
        payload = event.payload

        if et == "runtime.started":
            self._runtime["state"] = "running"
            self._runtime["details"] = payload
            self._runtime["supernodes"] = self._runtime_supernodes(payload)
            return

        if et == "runtime.stopped":
            self._runtime["state"] = "stopped"
            self._runtime["details"] = payload
            return

        if et == "supernodes.updated":
            nodes = payload.get("nodes", [])
            self._runtime["supernodes"] = nodes if isinstance(nodes, list) else []
            self._runtime["connection_name"] = payload.get("connection_name")
            self._runtime["flwr_home"] = payload.get("flwr_home")
            online = [n for n in self._runtime["supernodes"] if n.get("status") == "online"]
            self._runtime["online_count"] = len(online)
            self._runtime["total_count"] = len(self._runtime["supernodes"])
            if self._runtime.get("state") != "stopped":
                self._runtime["state"] = "running" if online else "idle"
            return

        if et == "run.started":
            self._run_status["status"] = "running"
            self._run_status["run"] = payload
            return

        if et == "run.status":
            self._run_status["status"] = payload.get("status", "running")
            self._run_status["last"] = payload
            return

        if et == "run.completed":
            self._run_status["status"] = "finished:completed"
            self._run_status["run"] = payload
            return

        if et == "run.failed":
            self._run_status["status"] = "finished:failed"
            self._run_status["run"] = payload
            return

        if et == "run.timeout":
            self._run_status["status"] = "timeout"
            self._run_status["run"] = payload
            return

        if et == "domain.started":
            if event.domain:
                self._domains.setdefault(event.domain, {})["status"] = "running"
                self._domains[event.domain]["started_at"] = event.ts_utc
            return

        if et == "domain.completed":
            if event.domain:
                self._domains.setdefault(event.domain, {})["status"] = "finished:completed"
                self._domains[event.domain]["completed_at"] = event.ts_utc
                self._domains[event.domain]["summary"] = payload
            return

        if et == "metrics.updated":
            domain = event.domain or str(payload.get("domain", ""))
            if not domain:
                return
            metrics = payload.get("metrics", payload)
            raw = metrics.get("raw_metrics", {})
            gated = metrics.get("gated_metrics", {})
            idx = self._quality_index.get(domain, 0)
            point = {
                "index": idx,
                "ts_utc": event.ts_utc,
                "raw": raw,
                "gated": gated,
                "unknown_threshold": metrics.get("unknown_threshold"),
            }
            self._quality.setdefault(domain, []).append(point)
            self._quality_index[domain] = idx + 1
            self._domains.setdefault(domain, {})["status"] = "finished:completed"
            self._domains[domain]["last_metrics_at"] = event.ts_utc
            return

        if et == "artifact.discovered":
            path = str(payload.get("path", ""))
            if path:
                self._artifacts[path] = payload

    @staticmethod
    def _event_dict(event: UiEventV1) -> dict[str, Any]:
        if hasattr(event, "model_dump"):
            return event.model_dump()  # type: ignore[no-any-return]
        return event.dict()  # type: ignore[no-any-return]

    @staticmethod
    def _runtime_supernodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
        pids = []
        processes = payload.get("processes")
        if isinstance(processes, dict):
            pids = processes.get("supernode_pids", [])
        num = payload.get("num_supernodes", len(pids))
        out = []
        for idx in range(int(num)):
            out.append(
                {
                    "id": idx,
                    "client_id": idx,
                    "pid": pids[idx] if idx < len(pids) else None,
                    "state": "running",
                }
            )
        return out
