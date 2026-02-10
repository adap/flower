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
"""Tests for event dispatcher."""


import time
from queue import Empty, Queue
from threading import Thread
from unittest.mock import MagicMock, Mock, patch

import grpc
import pytest

from flwr.proto.event_pb2 import Event, EventType
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.common.events import (
    EventDispatcher,
    _event_uploader,
    get_event_dispatcher,
    start_event_uploader,
    stop_event_uploader,
)


def test_event_dispatcher_subscribe() -> None:
    """Test that subscribe creates and returns a new queue."""
    # Prepare
    dispatcher = EventDispatcher()

    # Execute
    queue1 = dispatcher.subscribe()
    queue2 = dispatcher.subscribe()

    # Assert
    assert isinstance(queue1, Queue)
    assert isinstance(queue2, Queue)
    assert queue1 is not queue2
    assert len(dispatcher._subs) == 2


def test_event_dispatcher_unsubscribe() -> None:
    """Test that unsubscribe removes queue from subscribers."""
    # Prepare
    dispatcher = EventDispatcher()
    queue = dispatcher.subscribe()

    # Execute
    dispatcher.unsubscribe(queue)

    # Assert
    assert len(dispatcher._subs) == 0


def test_event_dispatcher_unsubscribe_nonexistent() -> None:
    """Test that unsubscribing a non-existent queue doesn't raise exception."""
    # Prepare
    dispatcher = EventDispatcher()
    queue: Queue[Event | None] = Queue()

    # Execute & Assert (should not raise)
    dispatcher.unsubscribe(queue)
    assert len(dispatcher._subs) == 0


def test_event_dispatcher_emit() -> None:
    """Test that emit sends event to all subscribers."""
    # Prepare
    dispatcher = EventDispatcher()
    queue1 = dispatcher.subscribe()
    queue2 = dispatcher.subscribe()
    event = Event(
        timestamp=time.time(),
        run_id=1,
        node_id=1,
        event_type=EventType.ROUND_STARTED,
    )

    # Execute
    dispatcher.emit(event)

    # Assert
    assert queue1.get_nowait() == event
    assert queue2.get_nowait() == event


def test_event_dispatcher_emit_no_subscribers() -> None:
    """Test that emit works with no subscribers."""
    # Prepare
    dispatcher = EventDispatcher()
    event = Event(
        timestamp=time.time(),
        node_id=1,
        run_id=1,
        event_type=EventType.ROUND_STARTED,
    )

    # Execute & Assert (should not raise)
    dispatcher.emit(event)


def test_event_dispatcher_emit_event() -> None:
    """Test that emit_event creates and broadcasts an event."""
    # Prepare
    dispatcher = EventDispatcher()
    queue = dispatcher.subscribe()

    # Execute
    dispatcher.emit_event(
        event_type=EventType.ROUND_STARTED,
        node_id=42,
        run_id=1,
        metadata={"number": "1"},
    )

    # Assert
    event = queue.get_nowait()
    assert event is not None
    assert event.node_id == 42
    assert event.event_type == EventType.ROUND_STARTED
    assert event.metadata["number"] == "1"
    assert event.timestamp > 0


def test_event_dispatcher_emit_event_no_metadata() -> None:
    """Test that emit_event works without metadata."""
    # Prepare
    dispatcher = EventDispatcher()
    queue = dispatcher.subscribe()

    # Execute
    dispatcher.emit_event(
        event_type=EventType.NODE_FIT_COMPLETED,
        run_id=1,
        node_id=10,
    )

    # Assert
    event = queue.get_nowait()
    assert event is not None
    assert event.node_id == 10
    assert event.event_type == EventType.NODE_FIT_COMPLETED
    assert len(event.metadata) == 0


def test_event_dispatcher_thread_safety() -> None:
    """Test that dispatcher handles concurrent subscribers safely."""
    # Prepare
    dispatcher = EventDispatcher()
    queues: list[Queue[Event | None]] = []

    def subscribe_worker() -> None:
        for _ in range(10):
            queues.append(dispatcher.subscribe())

    # Execute
    threads = [Thread(target=subscribe_worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    assert len(dispatcher._subs) == 50


def test_get_event_dispatcher_singleton() -> None:
    """Test that get_event_dispatcher returns the same instance."""
    # Execute
    dispatcher1 = get_event_dispatcher()
    dispatcher2 = get_event_dispatcher()

    # Assert
    assert dispatcher1 is dispatcher2


def test_get_event_dispatcher_thread_safety() -> None:
    """Test that get_event_dispatcher is thread-safe."""
    # Prepare
    dispatchers: list[EventDispatcher] = []

    def get_dispatcher_worker() -> None:
        dispatchers.append(get_event_dispatcher())

    # Execute
    threads = [Thread(target=get_dispatcher_worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert - all should be the same instance
    assert all(d is dispatchers[0] for d in dispatchers)


def test_event_uploader_batch_and_upload() -> None:
    """Test that uploader batches and uploads events."""
    # Prepare
    event_queue: Queue[Event | None] = Queue()
    stub = Mock(spec=ServerAppIoStub)
    stub.PushEvents = Mock()

    event1 = Event(
        timestamp=time.time(),
        node_id=1,
        run_id=1,
        event_type=EventType.ROUND_STARTED
    )
    event2 = Event(
        timestamp=time.time(),
        node_id=1,
        run_id=1,
        event_type=EventType.ROUND_COMPLETED
    )

    event_queue.put(event1)
    event_queue.put(event2)
    event_queue.put(None)  # Signal shutdown

    # Execute
    _event_uploader(event_queue, node_id=1, run_id=100, stub=stub)

    # Assert
    stub.PushEvents.assert_called_once()
    call_args = stub.PushEvents.call_args[0][0]
    assert call_args.HasField("node")
    assert call_args.node.node_id == 1
    assert call_args.run_id == 100
    assert len(call_args.events) == 2


def test_event_uploader_handles_unavailable_error() -> None:
    """Test that uploader ignores UNAVAILABLE gRPC errors."""
    # Prepare
    event_queue: Queue[Event | None] = Queue()
    stub = Mock(spec=ServerAppIoStub)

    error = grpc.RpcError()
    error.code = Mock(return_value=grpc.StatusCode.UNAVAILABLE)
    stub.PushEvents = Mock(side_effect=error)

    event = Event(timestamp=time.time(), node_id=1, event_type=EventType.ROUND_STARTED)
    event_queue.put(event)
    event_queue.put(None)  # Signal shutdown

    # Execute & Assert (should not raise)
    _event_uploader(event_queue, node_id=1, run_id=100, stub=stub)


def test_event_uploader_raises_other_grpc_errors() -> None:
    """Test that uploader raises non-UNAVAILABLE gRPC errors."""
    # Prepare
    event_queue: Queue[Event | None] = Queue()
    stub = Mock(spec=ServerAppIoStub)

    error = grpc.RpcError()
    error.code = Mock(return_value=grpc.StatusCode.INTERNAL)
    stub.PushEvents = Mock(side_effect=error)

    event = Event(timestamp=time.time(), node_id=1, event_type=EventType.ROUND_STARTED)
    event_queue.put(event)

    # Execute & Assert
    with pytest.raises(grpc.RpcError):
        _event_uploader(event_queue, node_id=1, run_id=100, stub=stub)


def test_event_uploader_empty_queue() -> None:
    """Test that uploader handles empty queue gracefully."""
    # Prepare
    event_queue: Queue[Event | None] = Queue()
    stub = Mock(spec=ServerAppIoStub)
    stub.PushEvents = Mock()

    event_queue.put(None)  # Immediate shutdown

    # Execute
    _event_uploader(event_queue, node_id=1, run_id=100, stub=stub)

    # Assert - should not have called PushEvents
    stub.PushEvents.assert_not_called()


@patch("flwr.common.events.time.sleep")
def test_event_uploader_waits_between_uploads(mock_sleep: MagicMock) -> None:
    """Test that uploader waits between upload cycles."""
    # Prepare
    event_queue: Queue[Event | None] = Queue()
    stub = Mock(spec=ServerAppIoStub)
    stub.PushEvents = Mock()

    # Put event, wait for one cycle, then shutdown
    event = Event(timestamp=time.time(), node_id=1, event_type=EventType.ROUND_STARTED)
    event_queue.put(event)

    def delayed_shutdown() -> None:
        time.sleep(0.1)
        event_queue.put(None)

    shutdown_thread = Thread(target=delayed_shutdown)
    shutdown_thread.start()

    # Execute
    _event_uploader(event_queue, node_id=1, run_id=100, stub=stub)
    shutdown_thread.join()

    # Assert - sleep should have been called
    assert mock_sleep.call_count >= 1


def test_start_event_uploader() -> None:
    """Test that start_event_uploader creates and starts thread."""
    # Prepare
    stub = Mock(spec=ServerAppIoStub)
    stub.PushEvents = Mock()

    # Execute
    thread, event_queue = start_event_uploader(node_id=1, run_id=100, stub=stub)

    try:
        # Assert
        assert isinstance(thread, Thread)
        assert isinstance(event_queue, Queue)
        assert thread.is_alive()
        assert thread.daemon is True
    finally:
        # Cleanup
        event_queue.put(None)
        thread.join(timeout=1.0)


def test_stop_event_uploader() -> None:
    """Test that stop_event_uploader gracefully stops the thread."""
    # Prepare
    stub = Mock(spec=ServerAppIoStub)
    stub.PushEvents = Mock()
    thread, event_queue = start_event_uploader(node_id=1, run_id=100, stub=stub)

    # Execute
    stop_event_uploader(event_queue, thread)

    # Assert
    assert not thread.is_alive()


def test_integration_dispatcher_to_uploader() -> None:
    """Test integration between dispatcher and uploader."""
    # Prepare
    stub = Mock(spec=ServerAppIoStub)
    stub.PushEvents = Mock()

    dispatcher = get_event_dispatcher()
    thread, event_queue = start_event_uploader(node_id=1, run_id=100, stub=stub)

    try:
        # Execute
        dispatcher.emit_event(
            event_type=EventType.ROUND_STARTED,
            node_id=1,
            metadata={"test": "integration"},
        )

        # Wait for event to be processed (EVENT_UPLOAD_INTERVAL is 0.5s)
        time.sleep(0.7)

        # Assert
        assert stub.PushEvents.call_count >= 1
        call_args = stub.PushEvents.call_args[0][0]
        assert call_args.HasField("node")
        assert call_args.run_id == 100
        assert len(call_args.events) >= 1
        assert call_args.events[0].event_type == EventType.ROUND_STARTED
    finally:
        # Cleanup
        stop_event_uploader(event_queue, thread)
        dispatcher.unsubscribe(event_queue)


def test_multiple_subscribers_receive_same_event() -> None:
    """Test that multiple subscribers receive the same event."""
    # Prepare
    dispatcher = EventDispatcher()
    queue1 = dispatcher.subscribe()
    queue2 = dispatcher.subscribe()
    queue3 = dispatcher.subscribe()

    # Execute
    dispatcher.emit_event(
        event_type=EventType.NODE_CONNECTED,
        node_id=5,
    )

    # Assert
    event1 = queue1.get_nowait()
    event2 = queue2.get_nowait()
    event3 = queue3.get_nowait()

    assert event1 is not None
    assert event2 is not None
    assert event3 is not None
    assert event1.event_type == EventType.NODE_CONNECTED
    assert event2.event_type == EventType.NODE_CONNECTED
    assert event3.event_type == EventType.NODE_CONNECTED
    assert event1.node_id == event2.node_id == event3.node_id == 5


def test_subscriber_can_be_removed_while_emitting() -> None:
    """Test that unsubscribing doesn't break ongoing emissions."""
    # Prepare
    dispatcher = EventDispatcher()
    queue1 = dispatcher.subscribe()
    queue2 = dispatcher.subscribe()

    # Execute
    dispatcher.emit_event(EventType.ROUND_STARTED, node_id=1)
    dispatcher.unsubscribe(queue1)
    dispatcher.emit_event(EventType.ROUND_COMPLETED, node_id=1)

    # Assert
    event1_1 = queue1.get_nowait()
    event2_1 = queue2.get_nowait()
    event2_2 = queue2.get_nowait()

    assert event1_1 is not None
    assert event2_1 is not None
    assert event2_2 is not None
    assert event1_1.event_type == EventType.ROUND_STARTED
    assert event2_1.event_type == EventType.ROUND_STARTED
    assert event2_2.event_type == EventType.ROUND_COMPLETED

    with pytest.raises(Empty):
        queue1.get_nowait()


def test_event_uploader_clears_events_after_upload() -> None:
    """Test that uploader clears events after successful upload."""
    # Prepare
    event_queue: Queue[Event | None] = Queue()
    stub = Mock(spec=ServerAppIoStub)
    stub.PushEvents = Mock()

    # Add multiple events
    for i in range(5):
        event = Event(
            timestamp=time.time(), node_id=i, event_type=EventType.ROUND_STARTED
        )
        event_queue.put(event)

    event_queue.put(None)

    # Execute
    _event_uploader(event_queue, node_id=1, run_id=100, stub=stub)

    # Assert - should have uploaded all events in one call
    assert stub.PushEvents.call_count == 1
    call_args = stub.PushEvents.call_args[0][0]
    assert call_args.HasField("node")
    assert len(call_args.events) == 5


def test_event_dispatcher_history_cleanup() -> None:
    """Test that event history is pruned when it exceeds max size."""
    # Prepare
    from flwr.common.constant import EVENT_HISTORY_MAX_SIZE

    dispatcher = EventDispatcher()

    # Emit more events than the maximum size
    num_events = EVENT_HISTORY_MAX_SIZE + 500
    for i in range(num_events):
        dispatcher.emit_event(
            event_type=EventType.ROUND_STARTED,
            node_id=i,
            run_id=1,
        )

    # Assert - history should be capped at max size
    assert len(dispatcher._events) == EVENT_HISTORY_MAX_SIZE

    # Verify that the oldest events were removed (should have events with higher IDs)
    first_event_node_id = dispatcher._events[0].node_id
    assert first_event_node_id >= 500  # First 500 should be pruned


def test_get_events_since_after_cleanup() -> None:
    """Test that get_events_since works correctly after history cleanup."""
    # Prepare
    from flwr.common.constant import EVENT_HISTORY_MAX_SIZE

    dispatcher = EventDispatcher()
    initial_timestamp = time.time()

    # Emit events that will trigger cleanup
    num_events = EVENT_HISTORY_MAX_SIZE + 100
    for i in range(num_events):
        dispatcher.emit_event(
            event_type=EventType.ROUND_STARTED,
            node_id=i,
            run_id=1,
        )

    # Get events after initial timestamp - should only return recent events
    events = dispatcher.get_events_since(initial_timestamp)

    # Assert - should get max size events (oldest ones were pruned)
    assert len(events) == EVENT_HISTORY_MAX_SIZE


def test_event_dispatcher_preserves_run_id() -> None:
    """Test that events preserve their run_id when emitted."""
    # Prepare
    dispatcher = EventDispatcher()
    queue = dispatcher.subscribe()

    # Execute - emit event with specific run_id
    dispatcher.emit_event(
        event_type=EventType.NODE_FIT_STARTED,
        node_id=42,
        run_id=123,
        metadata={"test": "value"},
    )

    # Assert
    event = queue.get_nowait()
    assert event is not None
    assert event.HasField("run_id") and event.run_id == 123
    assert event.node_id == 42
    assert event.metadata["test"] == "value"


def test_event_dispatcher_handles_optional_run_id() -> None:
    """Test that events can be emitted without run_id (None)."""
    # Prepare
    dispatcher = EventDispatcher()
    queue = dispatcher.subscribe()

    # Execute - emit event without run_id
    dispatcher.emit_event(
        event_type=EventType.NODE_CONNECTED,
        node_id=5,
        run_id=None,
    )

    # Assert
    event = queue.get_nowait()
    assert event is not None
    assert event.node_id == 5
    # run_id should not be set when None is passed
    assert not event.HasField("run_id")
