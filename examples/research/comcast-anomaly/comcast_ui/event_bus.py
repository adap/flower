"""Publish/subscribe event bus for websocket clients."""

from __future__ import annotations

import asyncio
import threading

from .schemas import UiEventV1


def _event_to_json(event: UiEventV1) -> str:
    if hasattr(event, "model_dump_json"):
        return event.model_dump_json()  # type: ignore[no-any-return]
    return event.json()  # type: ignore[no-any-return]


class EventBus:
    """Thread-safe broadcast bus backed by asyncio queues."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: set[asyncio.Queue[str]] = set()
        self._lock = threading.RLock()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=512)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        with self._lock:
            self._subs.discard(queue)

    def publish(self, event: UiEventV1) -> None:
        payload = _event_to_json(event)
        with self._lock:
            loop = self._loop
            queues = list(self._subs)
        if loop is None or not loop.is_running() or not queues:
            return
        for q in queues:
            loop.call_soon_threadsafe(self._safe_put_nowait, q, payload)

    @staticmethod
    def _safe_put_nowait(queue: asyncio.Queue[str], payload: str) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                _ = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                return
