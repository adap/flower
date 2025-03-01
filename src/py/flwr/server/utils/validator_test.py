# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Validator tests."""


import time
import unittest

from parameterized import parameterized

from flwr.common import DEFAULT_TTL, Error, Message, Metadata, RecordSet
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.proto import recordset_pb2
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes  # pylint: disable=E0611

from .validator import validate_message, validate_task_ins_or_res


def create_message(  # pylint: disable=R0913, R0917
    message_id: str = "",
    src_node_id: int = SUPERLINK_NODE_ID,
    dst_node_id: int = 456,
    ttl: int = DEFAULT_TTL,
    reply_to_message: str = "",
    has_content: bool = True,
    has_error: bool = False,
    msg_type: str = "mock",
) -> Message:
    """Create a Message for testing.

    By default, it creates a valid instruction message containing a RecordSet.
    """
    metadata = Metadata(
        run_id=0,
        message_id=message_id,
        src_node_id=src_node_id,
        dst_node_id=dst_node_id,
        reply_to_message=reply_to_message,
        group_id="",
        ttl=ttl,
        message_type=msg_type,
    )
    ret = Message(metadata=metadata, content=RecordSet())
    if not has_content:
        ret.__dict__["_content"] = None
    if has_error:
        ret.__dict__["_error"] = Error(0)
    return ret


class ValidatorTest(unittest.TestCase):
    """Test validation code in state."""

    def test_task_ins(self) -> None:
        """Test is_valid task_ins."""
        # Prepare
        # (consumer_node_id)
        valid_ins = [(1234), (4567)]
        invalid_ins = [(SUPERLINK_NODE_ID)]

        # Execute & Assert
        for consumer_node_id in valid_ins:
            msg = create_task_ins(consumer_node_id)
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for consumer_node_id in invalid_ins:
            msg = create_task_ins(consumer_node_id)
            val_errors = validate_task_ins_or_res(msg)
            self.assertTrue(val_errors)

    def test_is_valid_task_res(self) -> None:
        """Test is_valid task_res."""
        # Prepare
        # (producer_node_id, ancestry)
        valid_res: list[tuple[int, list[str]]] = [
            (1234, [str(SUPERLINK_NODE_ID)]),
        ]

        invalid_res: list[tuple[int, list[str]]] = [
            (SUPERLINK_NODE_ID, []),
            (SUPERLINK_NODE_ID, ["1234"]),
            (1234, []),
        ]

        # Execute & Assert
        for producer_node_id, ancestry in valid_res:
            msg = create_task_res(producer_node_id, ancestry)
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for producer_node_id, ancestry in invalid_res:
            msg = create_task_res(producer_node_id, ancestry)
            val_errors = validate_task_ins_or_res(msg)
            self.assertTrue(val_errors, (producer_node_id, ancestry))

    def test_task_ttl_expired(self) -> None:
        """Test validation for expired Task TTL."""
        # Prepare an expired TaskIns
        expired_task_ins = create_task_ins(0)
        expired_task_ins.task.created_at = time.time() - 10  # 10 seconds ago
        expired_task_ins.task.ttl = 6  # 6 seconds TTL

        expired_task_res = create_task_res(0, ["1"])
        expired_task_res.task.created_at = time.time() - 10  # 10 seconds ago
        expired_task_res.task.ttl = 6  # 6 seconds TTL

        # Execute & Assert
        val_errors_ins = validate_task_ins_or_res(expired_task_ins)
        self.assertIn("Task TTL has expired", val_errors_ins)

        val_errors_res = validate_task_ins_or_res(expired_task_res)
        self.assertIn("Task TTL has expired", val_errors_res)

    @parameterized.expand(  # type: ignore
        [
            # Valid messages
            (create_message(), False, False),
            (create_message(has_content=False, has_error=True), False, False),
            # `message_id` is set
            (create_message(message_id="123"), False, True),
            # `ttl` is zero
            (create_message(ttl=0), False, True),
            # `src_node_id` is not set
            (create_message(src_node_id=0), False, True),
            # `dst_node_id` is not set
            (create_message(dst_node_id=0), False, True),
            # `dst_node_id` is SUPERLINK
            (create_message(dst_node_id=SUPERLINK_NODE_ID), False, True),
            # `message_type` is not set
            (create_message(msg_type=""), False, True),
            # Both `content` and `error` are not set
            (create_message(has_content=False), False, True),
            # Both `content` and `error` are set
            (create_message(has_error=True), False, True),
            # `reply_to_message` is set in a non-reply message
            (create_message(reply_to_message="789"), False, True),
            # `reply_to_message` is not set in reply message
            (create_message(), True, True),
            # `dst_node_id` is not SuperLink in reply message
            (create_message(src_node_id=123, reply_to_message="blabla"), True, True),
        ]
    )
    def test_message(self, message: Message, is_reply: bool, should_fail: bool) -> None:
        """Test is_valid message."""
        # Execute
        val_errors = validate_message(message, is_reply_message=is_reply)

        # Assert
        if should_fail:
            self.assertTrue(val_errors)
        else:
            self.assertFalse(val_errors)


def create_task_ins(
    consumer_node_id: int,
    delivered_at: str = "",
) -> TaskIns:
    """Create a TaskIns for testing."""
    consumer = Node(
        node_id=consumer_node_id,
    )
    task = TaskIns(
        task_id="",
        group_id="",
        run_id=0,
        task=Task(
            delivered_at=delivered_at,
            producer=Node(node_id=SUPERLINK_NODE_ID),
            consumer=consumer,
            task_type="mock",
            recordset=recordset_pb2.RecordSet(),  # pylint: disable=E1101
            ttl=DEFAULT_TTL,
            created_at=time.time(),
        ),
    )

    return task


def create_task_res(
    producer_node_id: int,
    ancestry: list[str],
) -> TaskRes:
    """Create a TaskRes for testing."""
    task_res = TaskRes(
        task_id="",
        group_id="",
        run_id=0,
        task=Task(
            producer=Node(node_id=producer_node_id),
            consumer=Node(node_id=SUPERLINK_NODE_ID),
            ancestry=ancestry,
            task_type="mock",
            recordset=recordset_pb2.RecordSet(),  # pylint: disable=E1101
            ttl=DEFAULT_TTL,
            created_at=time.time(),
        ),
    )

    return task_res
