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
from flwr.proto.task_pb2 import SecureAggregation, Task, TaskIns, TaskRes
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
            msg = create_task_ins(
                consumer_node_id, anonymous, has_legacy_server_message=True
            )
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for consumer_node_id, anonymous in invalid_ins:
            msg = create_task_ins(
                consumer_node_id, anonymous, has_legacy_server_message=True
            )
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
            msg = create_task_res(
                producer_node_id, anonymous, ancestry, has_legacy_client_message=True
            )
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for producer_node_id, anonymous, ancestry in invalid_res:
            msg = create_task_res(
                producer_node_id, anonymous, ancestry, has_legacy_client_message=True
            )
            val_errors = validate_task_ins_or_res(msg)
            self.assertTrue(val_errors, (producer_node_id, anonymous, ancestry))

    def test_task_ins_secure_aggregation(self) -> None:
        """Test is_valid task_ins for Secure Aggregation."""
        # Prepare
        # (has_legacy_server_message, has_sa)
        valid_ins = [(True, True), (False, True)]
        invalid_ins = [(False, False)]

        # Execute & Assert
        for has_legacy_server_message, has_sa in valid_ins:
            msg = create_task_ins(1, False, has_legacy_server_message, has_sa)
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for has_legacy_server_message, has_sa in invalid_ins:
            msg = create_task_ins(1, False, has_legacy_server_message, has_sa)
            val_errors = validate_task_ins_or_res(msg)
            self.assertTrue(val_errors)

    def test_task_res_secure_aggregation(self) -> None:
        """Test is_valid task_res for Secure Aggregation."""
        # Prepare
        # (has_legacy_server_message, has_sa)
        valid_res = [(True, True), (False, True)]
        invalid_res = [(False, False)]

        # Execute & Assert
        for has_legacy_client_message, has_sa in valid_res:
            msg = create_task_res(0, True, ["1"], has_legacy_client_message, has_sa)
            val_errors = validate_task_ins_or_res(msg)
            self.assertFalse(val_errors)

        for has_legacy_client_message, has_sa in invalid_res:
            msg = create_task_res(0, True, ["1"], has_legacy_client_message, has_sa)
            val_errors = validate_task_ins_or_res(msg)
            self.assertTrue(val_errors)


def create_task_ins(
    consumer_node_id: int,
    anonymous: bool,
    has_legacy_server_message: bool = False,
    has_sa: bool = False,
    delivered_at: str = "",
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
            )
            if has_legacy_server_message
            else None,
            sa=SecureAggregation(named_values={}) if has_sa else None,
        ),
    )
    return task


def create_task_res(
    producer_node_id: int,
    anonymous: bool,
    ancestry: List[str],
    has_legacy_client_message: bool = False,
    has_sa: bool = False,
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
            )
            if has_legacy_client_message
            else None,
            sa=SecureAggregation(named_values={}) if has_sa else None,
        ),
    )
    return task_res
