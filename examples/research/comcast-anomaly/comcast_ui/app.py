"""FastAPI app for Comcast FL live UI."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except Exception as exc:  # pragma: no cover - only triggered when deps are missing
    _IMPORT_ERROR = exc
    FastAPI = None  # type: ignore[assignment,misc]
else:
    _IMPORT_ERROR = None

from .collectors.file_poller import FilePoller
from .collectors.hook_adapter import InMemoryUiHookSink
from .collectors.supernode_poller import SupernodePoller
from .event_bus import EventBus
from .panels.registry import PANEL_IDS, get_panel_spec, list_panel_specs
from .schemas import HealthResponse
from .state import UiStateStore


BASE_DIR = Path(__file__).resolve().parent


def _static_version() -> str:
    """Cache-busting version derived from static asset mtimes."""
    try:
        css_mtime = int((BASE_DIR / "static" / "styles.css").stat().st_mtime)
        js_mtime = int((BASE_DIR / "static" / "app.js").stat().st_mtime)
        return str(max(css_mtime, js_mtime))
    except FileNotFoundError:
        return "1"


def _panel_specs_payload() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for spec in list_panel_specs():
        if hasattr(spec, "model_dump"):
            payload.append(spec.model_dump())  # type: ignore[arg-type]
        else:
            payload.append(spec.dict())  # type: ignore[arg-type]
    return payload


def create_app(
    run_root: str | Path = "artifacts/fl",
    run_name: str = "default_run",
    domains: list[str] | None = None,
    poll_interval_sec: float = 1.0,
    supernode_poll_interval_sec: float = 2.0,
    superlink_connection: str | None = None,
    flwr_home: str | None = None,
) -> FastAPI:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI dependencies are required for comcast_ui. "
            "Install extras: pip install fastapi uvicorn jinja2"
        ) from _IMPORT_ERROR

    root_path = Path(run_root)
    effective_domains = list(domains or ["downstream_rxmer", "upstream_return"])

    app = FastAPI(title="Comcast FL Live UI", version="1.0")
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    store = UiStateStore()
    bus = EventBus()
    sink = InMemoryUiHookSink(store=store, bus=bus)
    poller = FilePoller(
        run_root=root_path,
        run_name=run_name,
        domains=effective_domains,
        sink=sink,
        interval_sec=poll_interval_sec,
    )
    supernode_poller = SupernodePoller(
        run_root=root_path,
        run_name=run_name,
        sink=sink,
        interval_sec=supernode_poll_interval_sec,
        connection_name=superlink_connection,
        flwr_home=flwr_home,
    )

    app.state.ui_store = store
    app.state.ui_bus = bus
    app.state.ui_sink = sink
    app.state.ui_poller = poller
    app.state.ui_supernode_poller = supernode_poller
    app.state.ui_run_root = root_path
    app.state.ui_run_name = run_name
    app.state.ui_domains = effective_domains

    @app.on_event("startup")
    async def _startup() -> None:
        bus.attach_loop(asyncio.get_running_loop())
        poller.start()
        supernode_poller.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        poller.stop()
        supernode_poller.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "run_name": run_name,
                "run_root": str(root_path),
                "panel_ids": PANEL_IDS,
                "static_version": _static_version(),
            },
        )

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        model = HealthResponse()
        if hasattr(model, "model_dump"):
            return model.model_dump()  # type: ignore[return-value]
        return model.dict()  # type: ignore[return-value]

    @app.get("/api/v1/layout")
    async def layout() -> dict[str, Any]:
        return {
            "run_name": run_name,
            "run_root": str(root_path),
            "panels": _panel_specs_payload(),
        }

    @app.get("/api/v1/state")
    async def state() -> dict[str, Any]:
        return store.get_snapshot()

    @app.get("/api/v1/panels/{panel_id}")
    async def panel(panel_id: str) -> dict[str, Any]:
        try:
            _ = get_panel_spec(panel_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown panel id: {panel_id}") from exc
        return store.get_panel_snapshot(panel_id)

    @app.websocket("/api/v1/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        queue = bus.subscribe()
        try:
            for evt in store.recent_events(limit=200):
                await ws.send_text(json.dumps(evt))
            while True:
                payload = await queue.get()
                await ws.send_text(payload)
        except WebSocketDisconnect:
            pass
        finally:
            bus.unsubscribe(queue)

    return app


def _env_domains() -> list[str]:
    raw = os.environ.get("COMCAST_UI_DOMAINS", "downstream_rxmer,upstream_return")
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts if parts else ["downstream_rxmer", "upstream_return"]


if FastAPI is not None:
    app = create_app(
        run_root=os.environ.get("COMCAST_UI_RUN_ROOT", "artifacts/fl"),
        run_name=os.environ.get("COMCAST_UI_RUN_NAME", "default_run"),
        domains=_env_domains(),
        poll_interval_sec=float(os.environ.get("COMCAST_UI_POLL_SEC", "1.0")),
        supernode_poll_interval_sec=float(os.environ.get("COMCAST_UI_SUPERNODE_POLL_SEC", "2.0")),
        superlink_connection=os.environ.get("COMCAST_UI_SUPERLINK_CONNECTION"),
        flwr_home=os.environ.get("COMCAST_UI_FLWR_HOME"),
    )
else:  # pragma: no cover - dependency-missing fallback
    app = None
