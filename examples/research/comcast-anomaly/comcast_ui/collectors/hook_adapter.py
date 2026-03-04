"""Hook sink adapter that converts FL hooks into UI events."""

from __future__ import annotations

from typing import Any

from comcast_fl.ui_hooks import UiHookSink

from ..event_bus import EventBus
from ..schemas import make_event
from ..state import UiStateStore


class InMemoryUiHookSink(UiHookSink):
    """Emit incoming runtime hooks into in-memory UI state and websocket bus."""

    def __init__(self, store: UiStateStore, bus: EventBus) -> None:
        self._store = store
        self._bus = bus

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        run_name: str | None = None,
        domain: str | None = None,
    ) -> None:
        event = make_event(event_type=event_type, payload=payload, run_name=run_name, domain=domain)
        self._store.apply_event(event)
        self._bus.publish(event)
