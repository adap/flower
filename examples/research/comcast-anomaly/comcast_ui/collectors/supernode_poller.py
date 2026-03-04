"""Poll live SuperNode status from Flower Control API via CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading

from comcast_fl.ui_hooks import UiHookSink


class SupernodePoller:
    """Poll `flwr supernode ls` and emit heartbeat-backed node status updates."""

    def __init__(
        self,
        run_root: Path,
        run_name: str,
        sink: UiHookSink,
        interval_sec: float = 2.0,
        connection_name: str | None = None,
        flwr_home: str | None = None,
    ) -> None:
        self._run_root = run_root
        self._run_name = run_name
        self._sink = sink
        self._interval = float(interval_sec)
        self._explicit_connection_name = connection_name
        self._explicit_flwr_home = flwr_home
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_signature: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="comcast-ui-supernode-poller",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(timeout=self._interval)

    def _poll_once(self) -> None:
        resolved = self._resolve_connection()
        if resolved is None:
            return
        connection_name, flwr_home = resolved

        env = None
        if flwr_home:
            env = {**os.environ, "FLWR_HOME": flwr_home}

        try:
            proc = subprocess.run(
                ["flwr", "supernode", "ls", connection_name, "--format", "json"],
                text=True,
                capture_output=True,
                check=False,
                env=env,
                timeout=10.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return
        if proc.returncode != 0:
            return
        try:
            payload = json.loads(proc.stdout)
        except Exception:
            return
        if not payload.get("success"):
            return

        nodes_raw = payload.get("nodes", [])
        nodes = [
            {
                "node_id": str(n.get("node-id")),
                "status": str(n.get("status", "unknown")),
                "owner_name": n.get("owner-name"),
                "online_at": n.get("online-at"),
                "offline_at": n.get("offline-at"),
                "online_elapsed": n.get("online-elapsed"),
            }
            for n in nodes_raw
            if n.get("node-id") is not None
        ]
        signature = json.dumps(
            {
                "connection_name": connection_name,
                "nodes": sorted(nodes, key=lambda x: x["node_id"]),
            },
            sort_keys=True,
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        self._sink.emit(
            "supernodes.updated",
            {
                "connection_name": connection_name,
                "flwr_home": flwr_home,
                "nodes": nodes,
            },
            run_name=self._run_name,
            domain=None,
        )

    def _resolve_connection(self) -> tuple[str, str | None] | None:
        if self._explicit_connection_name:
            return (self._explicit_connection_name, self._explicit_flwr_home)

        runtime_state = self._run_root / self._run_name / "deployment_runtime" / "runtime_state.json"
        if not runtime_state.exists():
            return None
        try:
            state = json.loads(runtime_state.read_text(encoding="utf-8"))
        except Exception:
            return None

        connection_name = state.get("connection_name")
        flwr_home = state.get("flwr_home")
        if not connection_name:
            return None
        return (str(connection_name), str(flwr_home) if flwr_home else None)
