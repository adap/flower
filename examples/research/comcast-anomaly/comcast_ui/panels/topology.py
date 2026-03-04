"""Federation topology panel payload builder."""

from __future__ import annotations

from typing import Any


def build_topology_payload(state: dict[str, Any]) -> dict[str, Any]:
    runtime = state.get("runtime", {})
    run_name = state.get("run_name")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    superlink_state = runtime.get("state", "idle")
    nodes.append({"id": "superlink", "type": "superlink", "state": superlink_state})

    supernodes = runtime.get("supernodes", [])
    for idx, node in enumerate(supernodes):
        source_id = node.get("node_id", node.get("id", idx))
        node_id = f"supernode-{source_id}"
        node_state = node.get("status", node.get("state", superlink_state))
        nodes.append(
            {
                "id": node_id,
                "type": "supernode",
                "state": node_state,
                "client_id": node.get("client_id", source_id),
                "node_id": source_id,
                "owner_name": node.get("owner_name"),
            }
        )
        edges.append({"source": "superlink", "target": node_id})

    return {
        "panel_id": "federation_topology",
        "run_name": run_name,
        "nodes": nodes,
        "edges": edges,
        "online_count": runtime.get("online_count"),
        "total_count": runtime.get("total_count"),
        "legend": {
            "states": ["online", "offline", "registered", "unregistered", "idle", "running", "stopped"]
        },
    }
