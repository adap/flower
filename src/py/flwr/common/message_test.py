# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Message tests."""

import time
from collections import namedtuple
from contextlib import ExitStack
from typing import Any, Callable, Dict, Optional

import pytest

# pylint: enable=E0611
from . import RecordSet
from .message import Error, Message, Metadata
from .serde_test import RecordMaker


@pytest.mark.parametrize(
    "content_fn, error_fn, context",
    [
        (
            lambda maker: maker.recordset(1, 1, 1),
            None,
            None,
        ),  # check when only content is set
        (None, lambda code: Error(code=code), None),  # check when only error is set
        (
            lambda maker: maker.recordset(1, 1, 1),
            lambda code: Error(code=code),
            pytest.raises(ValueError),
        ),  # check when both are set (ERROR)
        (None, None, pytest.raises(ValueError)),  # check when neither is set (ERROR)
    ],
)
def test_message_creation(
    content_fn: Callable[
        [
            RecordMaker,
        ],
        RecordSet,
    ],
    error_fn: Callable[[int], Error],
    context: Any,
) -> None:
    """Test Message creation attempting to pass content and/or error."""
    # Prepare
    maker = RecordMaker(state=2)
    metadata = maker.metadata()

    with ExitStack() as stack:
        if context:
            stack.enter_context(context)

        current_time = time.time()
        message = Message(
            metadata=metadata,
            content=None if content_fn is None else content_fn(maker),
            error=None if error_fn is None else error_fn(0),
        )

        assert message.metadata.created_at > current_time
        assert message.metadata.created_at < time.time()


def create_message_with_content(ttl: Optional[float] = None) -> Message:
    """Create a Message with content."""
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
    if ttl:
        metadata.ttl = ttl
    return Message(metadata=metadata, content=RecordSet())


def create_message_with_error(ttl: Optional[float] = None) -> Message:
    """Create a Message with error."""
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
    if ttl:
        metadata.ttl = ttl
    return Message(metadata=metadata, error=Error(code=1))


@pytest.mark.parametrize(
    "message_creation_fn",
    [
        create_message_with_content,
        create_message_with_error,
    ],
)
def test_altering_message(
    message_creation_fn: Callable[
        [],
        Message,
    ],
) -> None:
    """Test that a message with content doesn't allow setting an error.

    And viceversa.
    """
    message = message_creation_fn()

    with pytest.raises(ValueError):
        if message.has_content():
            message.error = Error(code=123)
        if message.has_error():
            message.content = RecordSet()


@pytest.mark.parametrize(
    "message_creation_fn,ttl,reply_ttl",
    [
        (create_message_with_content, 1e6, None),
        (create_message_with_error, 1e6, None),
        (create_message_with_content, 1e6, 3600),
        (create_message_with_error, 1e6, 3600),
    ],
)
def test_create_reply(
    message_creation_fn: Callable[
        [float],
        Message,
    ],
    ttl: float,
    reply_ttl: Optional[float],
) -> None:
    """Test reply creation from message."""
    message: Message = message_creation_fn(ttl)

    time.sleep(0.1)

    if message.has_error():
        dummy_error = Error(code=0, reason="it crashed")
        reply_message = message.create_error_reply(dummy_error, ttl=reply_ttl)
    else:
        reply_message = message.create_reply(content=RecordSet(), ttl=reply_ttl)

    # Ensure reply has a higher timestamp
    assert message.metadata.created_at < reply_message.metadata.created_at
    if reply_ttl:
        # Ensure the TTL is the one specify upon reply creation
        assert reply_message.metadata.ttl == reply_ttl
    else:
        # Ensure reply ttl is lower (since it uses remaining time left)
        assert message.metadata.ttl > reply_message.metadata.ttl

    assert message.metadata.src_node_id == reply_message.metadata.dst_node_id
    assert message.metadata.dst_node_id == reply_message.metadata.src_node_id
    assert reply_message.metadata.reply_to_message == message.metadata.message_id


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (
            Metadata,
            {
                "run_id": 123,
                "message_id": "msg_456",
                "src_node_id": 1,
                "dst_node_id": 2,
                "reply_to_message": "reply_789",
                "group_id": "group_xyz",
                "ttl": 10.0,
                "message_type": "request",
                "partition_id": None,
            },
        ),
        (Error, {"code": 1, "reason": "reason_098"}),
        (
            Message,
            {
                "metadata": RecordMaker(1).metadata(),
                "content": RecordMaker(1).recordset(1, 1, 1),
            },
        ),
        (
            Message,
            {
                "metadata": RecordMaker(2).metadata(),
                "error": Error(0, "some reason"),
            },
        ),
    ],
)
def test_repr(cls: type, kwargs: Dict[str, Any]) -> None:
    """Test string representations of Metadata/Message/Error."""
    # Prepare
    anon_cls = namedtuple(cls.__qualname__, kwargs.keys())  # type: ignore
    expected = anon_cls(**kwargs)
    actual = cls(**kwargs)

    # Assert
    assert str(actual) == str(expected)
