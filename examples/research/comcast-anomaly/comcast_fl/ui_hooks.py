"""Optional UI hook interface for runtime telemetry."""

from __future__ import annotations

from typing import Any, Protocol


class UiHookSink(Protocol):
    """Lightweight sink protocol for UI/telemetry event hooks."""

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        run_name: str | None = None,
        domain: str | None = None,
    ) -> None:
        """Emit one event payload."""


class NoOpUiHookSink:
    """Default no-op sink to preserve existing behavior when UI is disabled."""

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        run_name: str | None = None,
        domain: str | None = None,
    ) -> None:
        del event_type, payload, run_name, domain


def emit_hook(
    sink: UiHookSink | None,
    event_type: str,
    payload: dict[str, Any],
    run_name: str | None = None,
    domain: str | None = None,
) -> None:
    """Safely emit hook event if sink is configured."""
    if sink is None:
        return
    sink.emit(event_type=event_type, payload=payload, run_name=run_name, domain=domain)
