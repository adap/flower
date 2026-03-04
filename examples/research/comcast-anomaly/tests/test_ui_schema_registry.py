from __future__ import annotations

from comcast_ui.panels.registry import list_panel_specs
from comcast_ui.schemas import UiEventV1, make_event


def test_ui_event_v1_roundtrip() -> None:
    evt = make_event(
        event_type="metrics.updated",
        payload={"k": 1},
        run_name="demo",
        domain="downstream_rxmer",
    )
    assert isinstance(evt, UiEventV1)
    body = evt.model_dump()
    assert body["schema_version"] == "1.0"
    assert body["event_type"] == "metrics.updated"
    assert body["run_name"] == "demo"
    assert body["domain"] == "downstream_rxmer"


def test_panel_registry_contract() -> None:
    specs = list_panel_specs()
    ids = [s.id for s in specs]
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert ids == [
        "federation_topology",
        "round_timeline",
        "global_quality_trends",
        "unknown_gate_monitor",
        "client_participation",
        "non_iid_map",
        "update_divergence",
        "edge_constraints",
        "signal_gallery",
        "confusion_regime_explorer",
    ]
    implemented = [s.id for s in specs if s.implemented]
    assert implemented == [
        "federation_topology",
        "round_timeline",
        "global_quality_trends",
    ]
