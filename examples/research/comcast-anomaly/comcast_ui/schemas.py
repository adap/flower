"""Schemas for Comcast FL live UI events and payloads."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - fallback for envs without pydantic
    class BaseModel:  # type: ignore[no-redef]
        """Minimal BaseModel fallback used when pydantic is unavailable."""

        def __init__(self, **kwargs: Any) -> None:
            anns = getattr(self, "__annotations__", {})
            for name in anns:
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                elif hasattr(type(self), name):
                    default = getattr(type(self), name)
                    if isinstance(default, (dict, list, set)):
                        default = copy.deepcopy(default)
                    setattr(self, name, default)
                else:
                    setattr(self, name, None)

        def model_dump(self) -> dict[str, Any]:
            anns = getattr(self, "__annotations__", {})
            return {name: getattr(self, name) for name in anns}

        def dict(self) -> dict[str, Any]:
            return self.model_dump()

        def model_dump_json(self) -> str:
            return json.dumps(self.model_dump(), default=str)

        def json(self) -> str:
            return self.model_dump_json()

    def Field(default_factory=None, default=None):  # type: ignore[no-redef]
        if default_factory is not None:
            return default_factory()
        return default


UI_SCHEMA_VERSION = "1.0"


class UiEventV1(BaseModel):
    """Versioned event envelope for websocket/API transport."""

    schema_version: Literal["1.0"] = UI_SCHEMA_VERSION
    event_type: str
    ts_utc: str
    run_name: str | None = None
    domain: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PanelSpec(BaseModel):
    """Panel metadata for layout and compatibility checks."""

    id: str
    title: str
    category: str
    implemented: bool
    description: str
    data_contract_ref: str


class HealthResponse(BaseModel):
    """Simple health response."""

    ok: bool = True
    schema_version: Literal["1.0"] = UI_SCHEMA_VERSION


def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def make_event(
    event_type: str,
    payload: dict[str, Any],
    run_name: str | None = None,
    domain: str | None = None,
) -> UiEventV1:
    """Build a new UiEventV1 with a standardized timestamp."""
    return UiEventV1(
        event_type=event_type,
        ts_utc=utc_now_iso(),
        run_name=run_name,
        domain=domain,
        payload=payload,
    )
