"""Panel registry for Comcast FL live UI."""

from __future__ import annotations

from comcast_ui.schemas import PanelSpec


PANEL_SPECS: list[PanelSpec] = [
    PanelSpec(
        id="federation_topology",
        title="Federation Topology",
        category="federation",
        implemented=True,
        description="SuperLink/SuperNode/domain topology with lifecycle state.",
        data_contract_ref="comcast_ui/panels/topology.py::TopologyPayloadV1",
    ),
    PanelSpec(
        id="round_timeline",
        title="Round Timeline",
        category="federation",
        implemented=True,
        description="Chronological run and domain lifecycle status markers.",
        data_contract_ref="comcast_ui/panels/timeline.py::TimelinePayloadV1",
    ),
    PanelSpec(
        id="global_quality_trends",
        title="Global Quality Trends",
        category="quality",
        implemented=True,
        description="Domain-level raw and gated quality metrics over time.",
        data_contract_ref="comcast_ui/panels/quality.py::QualityPayloadV1",
    ),
    PanelSpec(
        id="unknown_gate_monitor",
        title="Unknown Gate Monitor",
        category="quality",
        implemented=False,
        description="Threshold behavior and unknown class forcing diagnostics.",
        data_contract_ref="comcast_ui/panels/stubs.py::UnknownGatePayloadV1",
    ),
    PanelSpec(
        id="client_participation",
        title="Client Participation",
        category="federation",
        implemented=False,
        description="Per-round client sampling and participation stability.",
        data_contract_ref="comcast_ui/panels/stubs.py::ClientParticipationPayloadV1",
    ),
    PanelSpec(
        id="non_iid_map",
        title="Non-IID Map",
        category="data",
        implemented=False,
        description="Inter-client heterogeneity by class/regime/context/template axes.",
        data_contract_ref="comcast_ui/panels/stubs.py::NonIidMapPayloadV1",
    ),
    PanelSpec(
        id="update_divergence",
        title="Update Divergence",
        category="federation",
        implemented=False,
        description="Distance between client model updates and aggregate direction.",
        data_contract_ref="comcast_ui/panels/stubs.py::UpdateDivergencePayloadV1",
    ),
    PanelSpec(
        id="edge_constraints",
        title="Edge Constraints",
        category="edge",
        implemented=False,
        description="Edge gate status for params, latency proxy, and quant readiness.",
        data_contract_ref="comcast_ui/panels/stubs.py::EdgeConstraintsPayloadV1",
    ),
    PanelSpec(
        id="signal_gallery",
        title="Signal Gallery",
        category="workload",
        implemented=False,
        description="Representative static/sequence synthetic signal slices by class/domain.",
        data_contract_ref="comcast_ui/panels/stubs.py::SignalGalleryPayloadV1",
    ),
    PanelSpec(
        id="confusion_regime_explorer",
        title="Confusion/Regime Explorer",
        category="quality",
        implemented=False,
        description="Class confusion and per-regime behavior explorer payloads.",
        data_contract_ref="comcast_ui/panels/stubs.py::ConfusionRegimePayloadV1",
    ),
]


PANEL_IDS = [p.id for p in PANEL_SPECS]


def list_panel_specs() -> list[PanelSpec]:
    return list(PANEL_SPECS)


def get_panel_spec(panel_id: str) -> PanelSpec:
    found = next((p for p in PANEL_SPECS if p.id == panel_id), None)
    if found is None:
        raise KeyError(panel_id)
    return found
