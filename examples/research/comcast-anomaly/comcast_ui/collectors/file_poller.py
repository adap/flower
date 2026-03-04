"""Artifact and runtime-state polling collector for UI telemetry."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from comcast_fl.ui_hooks import UiHookSink


class FilePoller:
    """Poll FL artifact files and emit inferred UI events."""

    def __init__(
        self,
        run_root: Path,
        run_name: str,
        domains: list[str],
        sink: UiHookSink,
        interval_sec: float = 1.0,
    ) -> None:
        self._run_root = run_root
        self._run_name = run_name
        self._domains = list(domains)
        self._sink = sink
        self._interval = float(interval_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_mtime_ns: dict[Path, int] = {}
        self._runtime_started = False
        self._runtime_stopped = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="comcast-ui-file-poller", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._scan_once()
            self._stop.wait(timeout=self._interval)

    def _scan_once(self) -> None:
        base = self._run_root / self._run_name
        runtime_state = base / "deployment_runtime" / "runtime_state.json"
        summary_path = base / "summary.json"
        self._check_path(runtime_state, artifact_type="runtime_state", domain=None)
        self._check_path(summary_path, artifact_type="summary", domain=None)
        for domain in self._domains:
            metrics_path = base / domain / "metrics.json"
            self._check_path(metrics_path, artifact_type="metrics", domain=domain)

    def _check_path(self, path: Path, artifact_type: str, domain: str | None) -> None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return

        mtime = int(stat.st_mtime_ns)
        prev = self._seen_mtime_ns.get(path)
        if prev is not None and prev == mtime:
            return
        self._seen_mtime_ns[path] = mtime

        payload = {
            "artifact_type": artifact_type,
            "path": str(path),
            "mtime_ns": mtime,
            "size": int(stat.st_size),
        }
        self._sink.emit("artifact.discovered", payload, run_name=self._run_name, domain=domain)

        try:
            raw = path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except Exception:
            return

        if artifact_type == "runtime_state":
            if not self._runtime_started and "started_at" in doc:
                self._runtime_started = True
                self._sink.emit("runtime.started", doc, run_name=self._run_name, domain=None)
            if not self._runtime_stopped and "stopped_at" in doc:
                self._runtime_stopped = True
                self._sink.emit("runtime.stopped", doc, run_name=self._run_name, domain=None)
            return

        if artifact_type == "metrics" and domain:
            self._sink.emit(
                "metrics.updated",
                {"metrics": doc, "path": str(path), "domain": domain},
                run_name=self._run_name,
                domain=domain,
            )
            self._sink.emit(
                "domain.completed",
                {"status": "finished:completed", "path": str(path)},
                run_name=self._run_name,
                domain=domain,
            )
            return

        if artifact_type == "summary":
            self._sink.emit(
                "run.completed",
                {"summary": doc, "path": str(path)},
                run_name=self._run_name,
                domain=None,
            )
