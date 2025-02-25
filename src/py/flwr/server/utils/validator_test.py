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
from typing import Optional

from parameterized import parameterized

from flwr.common import DEFAULT_TTL, Error, Message, Metadata, RecordSet
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.proto import recordset_pb2
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes  # pylint: disable=E0611

from .validator import validate_message, validate_task_ins_or_res


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
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "",
                RecordSet(),
                None,
                "mock",
                False,
                False,
            ),  # Should pass
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "",
                None,
                Error(0),
                "mock",
                False,
                False,
            ),  # Should pass
            (
                "123",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "",
                RecordSet(),
                None,
                "mock",
                False,
                True,
            ),  # message_id is set
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                0,
                "",
                RecordSet(),
                None,
                "mock",
                False,
                True,
            ),  # ttl is zero
            (
                "",
                None,
                456,
                DEFAULT_TTL,
                "",
                RecordSet(),
                None,
                "mock",
                False,
                True,
            ),  # unset src_node_id
            (
                "",
                SUPERLINK_NODE_ID,
                None,
                DEFAULT_TTL,
                "",
                RecordSet(),
                None,
                "mock",
                False,
                True,
            ),  # unset dst_node_id
            (
                "",
                SUPERLINK_NODE_ID,
                SUPERLINK_NODE_ID,
                DEFAULT_TTL,
                "",
                RecordSet(),
                None,
                "mock",
                False,
                True,
            ),  # dst_node_id is SUPERLINK
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "",
                RecordSet(),
                None,
                "",
                False,
                True,
            ),  # message_type unset
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "",
                None,
                None,
                "mock",
                False,
                True,
            ),  # message has both content and error unset
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "",
                RecordSet(),
                Error(0),
                "mock",
                False,
                True,
            ),  # message has both content and error set
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "789",
                RecordSet(),
                None,
                "mock",
                False,
                True,
            ),  # reply_to_message is set it's not a reply message
            (
                "",
                SUPERLINK_NODE_ID,
                456,
                DEFAULT_TTL,
                "",
                RecordSet(),
                None,
                "mock",
                True,
                True,
            ),  # reply_to_message isn't set in reply message
            (
                "",
                123,
                456,
                DEFAULT_TTL,
                "blabla",
                RecordSet(),
                None,
                "mock",
                True,
                True,
            ),  # is reply message but dst_node_id isn't superlink
        ]
    )
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def test_message(
        self,
        message_id: str,
        src_node_id: Optional[int],
        dst_node_id: Optional[int],
        ttl: int,
        reply_to_message: str,
        content: Optional[RecordSet],
        error: Optional[Error],
        msg_type: str,
        is_reply: bool,
        should_fail: bool,
    ) -> None:
        """Test is_valid message."""
        metadata = Metadata(
            run_id=0,
            message_id=message_id,
            src_node_id=src_node_id,  # type: ignore
            dst_node_id=dst_node_id,  # type: ignore
            reply_to_message=reply_to_message,
            group_id="",
            ttl=ttl,
            message_type=msg_type,
        )

        if content is None and error is None:
            message = Message(metadata=metadata, content=RecordSet())
            # pylint: disable-next=protected-access
            message._content = None  # type: ignore
        elif content is not None and error is not None:
            message = Message(metadata=metadata, content=content)
            # pylint: disable-next=protected-access
            message._error = error  # type: ignore
        else:
            # Normal creation
            message = Message(metadata=metadata, content=content, error=error)

        # Execute & Assert
        val_errors = validate_message(message, is_reply_message=is_reply)
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
            recordset=recordset_pb2.RecordSet(  # pylint: disable=E1101
                parameters={}, metrics={}, configs={}
            ),
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
            recordset=recordset_pb2.RecordSet(  # pylint: disable=E1101
                parameters={}, metrics={}, configs={}
            ),
            ttl=DEFAULT_TTL,
            created_at=time.time(),
        ),
    )

    return task_res
