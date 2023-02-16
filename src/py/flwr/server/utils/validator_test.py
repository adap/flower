# Copyright 2023 Adap GmbH. All Rights Reserved.
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
import unittest
from typing import List, Tuple

from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage

from .validator import validate_task_ins_or_res


class ValidatorTest(unittest.TestCase):
    """Test validation code in state."""

    def test_task_ins(self) -> None:
        """Test is_valid task_ins."""
        # Prepare
        # (consumer_node_id, anonymous)
        valid_ins = [(0, True), (1, False)]
        invalid_ins = [(0, False), (1, True)]

        # Execute & Assert
        for consumer_node_id, anonymous in valid_ins:
            msg = create_task_ins(consumer_node_id, anonymous)
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for consumer_node_id, anonymous in invalid_ins:
            msg = create_task_ins(consumer_node_id, anonymous)
            val_errors = validate_task_ins_or_res(msg)
            self.assertTrue(val_errors)

    def test_is_valid_task_res(self) -> None:
        """Test is_valid task_res."""
        # Prepare
        # (producer_node_id, anonymous, ancestry)
        valid_res: List[Tuple[int, bool, List[str]]] = [
            (0, True, ["1"]),
            (1, False, ["1"]),
        ]

        invalid_res: List[Tuple[int, bool, List[str]]] = [
            (0, False, []),
            (0, False, ["1"]),
            (0, True, []),
            (1, False, []),
            (1, True, []),
            (1, True, ["1"]),
        ]

        # Execute & Assert
        for producer_node_id, anonymous, ancestry in valid_res:
            msg = create_task_res(producer_node_id, anonymous, ancestry)
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for producer_node_id, anonymous, ancestry in invalid_res:
            msg = create_task_res(producer_node_id, anonymous, ancestry)
            val_errors = validate_task_ins_or_res(msg)
            self.assertTrue(val_errors, (producer_node_id, anonymous, ancestry))


def create_task_ins(
    consumer_node_id: int, anonymous: bool, delivered_at: str = ""
) -> TaskIns:
    """Create a TaskIns for testing."""
    consumer = Node(
        node_id=consumer_node_id,
        anonymous=anonymous,
    )
    task = TaskIns(
        task_id="",
        group_id="",
        workload_id="",
        task=Task(
            delivered_at=delivered_at,
            producer=Node(node_id=0, anonymous=True),
            consumer=consumer,
            legacy_server_message=ServerMessage(
                reconnect_ins=ServerMessage.ReconnectIns()
            ),
        ),
    )
    return task


def create_task_res(
    producer_node_id: int, anonymous: bool, ancestry: List[str]
) -> TaskRes:
    """Create a TaskRes for testing."""
    task_res = TaskRes(
        task_id="",
        group_id="",
        workload_id="",
        task=Task(
            producer=Node(node_id=producer_node_id, anonymous=anonymous),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=ancestry,
            legacy_client_message=ClientMessage(
                disconnect_res=ClientMessage.DisconnectRes()
            ),
        ),
    )
    return task_res
