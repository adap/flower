"""Round timeline panel payload builder."""

from __future__ import annotations

from typing import Any


_TIMELINE_EVENT_TYPES = {
    "runtime.started",
    "runtime.stopped",
    "run.started",
    "run.status",
    "run.completed",
    "run.failed",
    "run.timeout",
    "domain.started",
    "domain.completed",
}


def build_timeline_payload(state: dict[str, Any]) -> dict[str, Any]:
    timeline = [e for e in state.get("timeline", []) if e.get("event_type") in _TIMELINE_EVENT_TYPES]

    durations: list[dict[str, Any]] = []
    started_by_domain: dict[str, str] = {}
    for event in timeline:
        event_type = str(event.get("event_type", ""))
        domain = str(event.get("domain", ""))
        ts = str(event.get("ts_utc", ""))
        if event_type == "domain.started" and domain:
            started_by_domain[domain] = ts
        if event_type == "domain.completed" and domain:
            durations.append({"domain": domain, "started_at": started_by_domain.get(domain), "ended_at": ts})

    return {
        "panel_id": "round_timeline",
        "run_name": state.get("run_name"),
        "events": timeline[-200:],
        "durations": durations,
    }
