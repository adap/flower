# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


from contextlib import ExitStack
from typing import Any, Callable

import pytest

# pylint: enable=E0611
from . import RecordSet
from .message import Error, Message
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

        _ = Message(
            metadata=metadata,
            content=None if content_fn is None else content_fn(maker),
            error=None if error_fn is None else error_fn(0),
        )


def create_message_with_content() -> Message:
    """Create a Message with content."""
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
    return Message(metadata=metadata, content=RecordSet())


def create_message_with_error() -> Message:
    """Create a Message with error."""
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
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
