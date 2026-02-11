# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Event dispatcher for real-time training events."""

import time
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Protocol

import grpc

from flwr.common.constant import EVENT_HISTORY_MAX_SIZE, EVENT_UPLOAD_INTERVAL
from flwr.proto.event_pb2 import Event, EventType, PushEventsRequest, PushEventsResponse
from flwr.proto.node_pb2 import Node


class EventStub(Protocol):
    """Protocol for stubs that support PushEvents."""

    def PushEvents(
        self, request: PushEventsRequest, timeout: float | None = None
    ) -> PushEventsResponse:
        """Push events to the server."""
        ...


class EventDispatcher:
    """Thread-safe event dispatcher for real-time federated learning events.

    The EventDispatcher implements a publish-subscribe pattern that allows multiple
    subscribers to receive training events as they occur. All operations are
    thread-safe and events are delivered to all active subscribers.
    """

    def __init__(self) -> None:
        self._subs: list[Queue[Event | None]] = []
        self._events: list[Event] = []
        self._lock = Lock()

    def subscribe(self) -> Queue[Event | None]:
        """Register a new subscriber to receive events.

        Creates a new queue for the subscriber and adds it to the list of active
        subscribers. The queue will receive all events emitted after subscription.

        Returns
        -------
        Queue[Event | None]
            A queue that will receive Event objects. A None value signals shutdown.

        Notes
        -----
        Subscribers must call `unsubscribe` to prevent memory leaks when they no
        longer need to receive events.
        """
        queue: Queue[Event | None] = Queue()

        with self._lock:
            self._subs.append(queue)
        return queue

    def unsubscribe(self, queue: Queue[Event | None]) -> None:
        """Remove a subscriber from the event dispatcher.

        Parameters
        ----------
        queue : Queue[Event | None]
            The queue previously returned by `subscribe` that should be removed.

        Notes
        -----
        If the queue is not found in the subscriber list, this method silently
        succeeds without raising an exception.
        """
        with self._lock:
            try:
                self._subs.remove(queue)
            except ValueError:
                pass

    def emit(self, event: Event) -> None:
        """Broadcast an event to all active subscribers.

        The event is delivered to all subscribers without blocking. If a subscriber's
        queue is full, this method will raise an exception.

        Parameters
        ----------
        event : Event
            The event object to broadcast to all subscribers.

        Notes
        -----
        This method uses `put_nowait` to avoid blocking. Subscribers should
        consume events promptly to prevent queue overflow.

        The event history is automatically pruned when it exceeds
        EVENT_HISTORY_MAX_SIZE to prevent unbounded memory growth.
        """
        with self._lock:
            self._events.append(event)

            if len(self._events) > EVENT_HISTORY_MAX_SIZE:
                self._events = self._events[-EVENT_HISTORY_MAX_SIZE:]

            for queue in self._subs:
                queue.put_nowait(event)

    def emit_event(
        self,
        event_type: EventType.ValueType,
        node_id: int = 0,
        run_id: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Create and broadcast an event with the current timestamp.

        This is a convenience method that constructs an Event object and emits it
        to all subscribers in a single operation.

        Parameters
        ----------
        event_type : EventType.ValueType
            The type of event to emit (e.g., ROUND_STARTED, NODE_FIT_COMPLETED).
        node_id : int (default: 0)
            The ID of the node associated with this event.
        metadata : Optional[Dict[str, str]] (default: None)
            Additional key-value pairs providing context about the event.
        """
        event = Event(
            run_id=run_id,
            timestamp=time.time(),
            node_id=node_id,
            event_type=event_type,
        )
        if metadata:
            event.metadata.update(metadata)

        self.emit(event)

    def get_events_since(self, after_timestamp: float) -> list[Event]:
        """Retrieve all events that occurred after the specified timestamp.

        Parameters
        ----------
        after_timestamp : float
            The timestamp (in seconds since epoch) after which to retrieve events.

        Returns
        -------
        list[Event]
            A list of events that occurred after the specified timestamp, ordered
            by timestamp.

        Notes
        -----
        This method is thread-safe and returns a copy of the matching events.
        The internal event history grows unbounded, so in production systems
        you may want to implement cleanup of old events.
        """
        with self._lock:
            return [evt for evt in self._events if evt.timestamp > after_timestamp]


_event_dispatcher: EventDispatcher | None = None
_dispatcher_lock = Lock()


def get_event_dispatcher() -> EventDispatcher:
    """Get or create the global EventDispatcher singleton instance.

    This function implements thread-safe lazy initialization of the global
    event dispatcher using double-checked locking pattern.

    Returns
    -------
    EventDispatcher
        The global EventDispatcher instance shared across the application.

    Notes
    -----
    The double-checked locking pattern ensures that only one instance is
    created even in multi-threaded environments, while minimizing lock
    contention after initialization.
    """
    global _event_dispatcher

    if _event_dispatcher is None:
        with _dispatcher_lock:
            if _event_dispatcher is None:
                _event_dispatcher = EventDispatcher()

    return _event_dispatcher


def _event_uploader(
    event_queue: Queue[Event | None],
    node_id: int,
    run_id: int,
    stub: EventStub,
) -> None:
    """Background worker that batches and uploads events to the server.

    This function runs in a daemon thread, continuously polling the event queue,
    batching events, and uploading them to the ServerAppIo service at regular
    intervals. It gracefully handles network errors and shutdown signals.

    Parameters
    ----------
    event_queue : Queue[Event | None]
        The queue from which to consume events. A None value signals shutdown.
    node_id : int
        The ID of the node generating these events.
    run_id : int
        The ID of the federated learning run.
    stub : ServerAppIoStub
        The gRPC stub for communicating with the ServerAppIo service.

    Notes
    -----
    The uploader batches events to reduce network overhead and uploads them
    every `EVENT_UPLOAD_INTERVAL` seconds. Network errors with status code
    UNAVAILABLE are silently ignored to handle transient connectivity issues,
    but other errors will propagate and terminate the thread.
    """
    exit_flag = False
    node = Node(node_id=node_id)
    events: list[Event] = []
    while True:
        # Fetch all events from the queue
        try:
            while True:
                evt = event_queue.get_nowait()
                if evt is None:
                    exit_flag = True
                    break
                events.append(evt)
        except Empty:
            pass

        # Upload if any events
        if events:
            req = PushEventsRequest(
                node=node,
                run_id=run_id,
                events=events,
            )
            try:
                stub.PushEvents(req)
                events.clear()
            except grpc.RpcError as e:
                # Ignore minor network errors
                # pylint: disable-next=no-member
                if e.code() != grpc.StatusCode.UNAVAILABLE:
                    raise e

        if exit_flag:
            break

        time.sleep(EVENT_UPLOAD_INTERVAL)


def start_event_uploader(
    node_id: int,
    run_id: int,
    stub: EventStub,
) -> tuple[Thread, Queue[Event | None]]:
    """Start a background thread that uploads events to the server.

    This function subscribes to the global event dispatcher and starts a daemon
    thread that will continuously upload events to the ServerAppIo service.

    Parameters
    ----------
    node_id : int
        The ID of the node generating events.
    run_id : int
        The ID of the federated learning run.
    stub : ServerAppIoStub
        The gRPC stub for communicating with the ServerAppIo service.

    Returns
    -------
    thread : Thread
        The daemon thread running the event uploader.
    event_queue : Queue[Event | None]
        The queue subscribed to the dispatcher. Pass this to `stop_event_uploader`
        for graceful shutdown.

    Notes
    -----
    The returned thread is a daemon thread and will not prevent the process from
    exiting. Call `stop_event_uploader` to ensure all events are uploaded before
    the application terminates.
    """
    dispatcher = get_event_dispatcher()
    event_queue: Queue[Event | None] = dispatcher.subscribe()
    thread = Thread(
        target=_event_uploader,
        args=(event_queue, node_id, run_id, stub),
        daemon=True,
    )
    thread.start()
    return thread, event_queue


def stop_event_uploader(
    event_queue: Queue[Event | None],
    event_uploader: Thread,
) -> None:
    """Gracefully stop the event uploader thread.

    Sends a shutdown signal to the uploader thread and waits for it to finish
    uploading any remaining events.

    Parameters
    ----------
    event_queue : Queue[Event | None]
        The event queue returned by `start_event_uploader`.
    event_uploader : Thread
        The uploader thread returned by `start_event_uploader`.

    Notes
    -----
    This function sends a None value through the queue to signal shutdown, then
    waits up to 5 seconds for the thread to finish. If the thread does not
    complete within the timeout, it will be abandoned (as it is a daemon thread).
    """
    event_queue.put(None)
    event_uploader.join(timeout=5.0)
