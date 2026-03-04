"""Stub payloads for not-yet-implemented panels."""

from __future__ import annotations

from typing import Any


STUB_SCHEMA_SNIPPETS: dict[str, dict[str, Any]] = {
    "unknown_gate_monitor": {
        "threshold": "float",
        "unknown_rate": "float",
        "raw_vs_gated_delta": "dict[str, float]",
    },
    "client_participation": {
        "rounds": "list[int]",
        "sampled_clients": "dict[str, list[str]]",
    },
    "non_iid_map": {
        "axis_severity": "dict[str, float]",
        "client_divergence": "list[float]",
    },
    "update_divergence": {
        "rounds": "list[int]",
        "l2_distance": "list[float]",
        "cosine_to_global": "list[float]",
    },
    "edge_constraints": {
        "params": "int",
        "p95_latency_ms_cpu_proxy": "float",
        "quantization_ready": "bool",
        "pass_edge_gate": "bool",
    },
    "signal_gallery": {
        "domain": "str",
        "class_examples": "dict[str, list[float]]",
    },
    "confusion_regime_explorer": {
        "confusion_matrix": "list[list[int]]",
        "regime_macro_f1": "dict[str, float]",
    },
}


def stub_panel_payload(panel_id: str) -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "status": "stub",
        "message": "Not yet implemented in this pass.",
        "expected_schema": STUB_SCHEMA_SNIPPETS.get(panel_id, {}),
        "plot": None,
    }
