# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for InflatableObject helpers to communicate with gRPC servicers."""


import unittest
from typing import Union
from unittest.mock import Mock

import numpy as np
from parameterized import parameterized

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611

from .inflatable import get_desdendant_object_ids
from .inflatable_grpc_utils import pull_object_from_servicer, push_object_to_servicer

base_cases = [
    ({"a": ConfigRecord({"a": 123, "b": 123})}, 1),  # Single w/o children
    (
        {
            "a": ConfigRecord({"a": 123, "b": 123}),
            "b": ConfigRecord({"a": 123, "b": 123}),
        },
        1,
    ),  # Two identical
    (
        {"a": ConfigRecord({"a": 123, "b": 123}), "b": ConfigRecord()},
        2,
    ),  # Different
    (
        {
            "a": ConfigRecord({"a": 123, "b": 123}),
            "b": ArrayRecord([np.array([1, 2]), np.array([3, 4])]),
        },
        4,
    ),  # Mixed with children
]


class TestInflatableStubHelpers(unittest.TestCase):  # pylint: disable=R0902
    """Test helpers to push and pull InflatableObjects."""

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        self.mock_store: dict[str, bytes] = {}
        self.mock_stub = Mock()

        def push_object(request: PushObjectRequest) -> PushObjectResponse:
            self.mock_store[request.object_id] = request.object_content
            return PushObjectResponse()

        def pull_object(request: PullObjectRequest) -> PullObjectResponse:
            return PullObjectResponse(object_content=self.mock_store[request.object_id])

        self.mock_stub.PushObject.side_effect = push_object
        self.mock_stub.PullObject.side_effect = pull_object

    @parameterized.expand(base_cases)  # type: ignore
    def test_push_object_with_helper_function(
        self,
        records: dict[str, Union[ArrayRecord, ConfigRecord, MetricRecord]],
        expected_obj_count: int,
    ) -> None:
        """Use helper function to push an object recursively."""
        # Prepare
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        node = Node(node_id=456)
        run_id = 1234
        # +2 due to the RecordDict and Message
        expected_obj_count += 2

        # Execute
        pushed_object_ids = push_object_to_servicer(obj, self.mock_stub, node, run_id)

        # Assert
        # Expected number of objects were pushed
        assert self.mock_stub.PushObject.call_count == expected_obj_count
        assert len(self.mock_store) == expected_obj_count
        assert len(pushed_object_ids) == expected_obj_count

    def test_push_objects_filtering_by_obj_ids_list(self) -> None:
        """Test pushing objects based on list of object_ids to push."""
        # Prepare
        records: dict[str, Union[ArrayRecord, ConfigRecord, MetricRecord]] = {
            "a": ConfigRecord({"a": 123, "b": 123}),
            "b": ArrayRecord([np.array([1, 2]), np.array([3, 4])]),
        }
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        node = Node(node_id=456)
        run_id = 1234

        # Compute descendants
        descendants = list(get_desdendant_object_ids(obj))
        # Take first two
        obj_to_push = set([obj.object_id] + descendants[:2])
        expected_obj_count = 3

        # Execute
        pushed_object_ids = push_object_to_servicer(
            obj, self.mock_stub, node, run_id, obj_to_push
        )

        # Assert
        # Expected number of objects were pushed
        assert self.mock_stub.PushObject.call_count == expected_obj_count
        assert len(self.mock_store) == expected_obj_count
        assert len(pushed_object_ids) == expected_obj_count
        assert obj_to_push == pushed_object_ids

    @parameterized.expand(base_cases)  # type: ignore
    def test_pull_object_with_helper_function(
        self,
        records: dict[str, Union[ArrayRecord, ConfigRecord, MetricRecord]],
        expected_obj_count: int,
    ) -> None:
        """Use helper function to pull an object recursively."""
        # Prepare
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        node = Node(node_id=456)
        run_id = 1234
        # +2 due to the RecordDict and Message
        expected_obj_count += 2

        # Execute
        push_object_to_servicer(obj, self.mock_stub, node, run_id)
        pulled_obj = pull_object_from_servicer(
            obj.object_id, self.mock_stub, node, run_id
        )

        # Assert
        # Expected number of objects were pulled
        assert self.mock_stub.PullObject.call_count == expected_obj_count
        assert pulled_obj.object_id == obj.object_id
        self.assertNotEqual(pulled_obj, obj)
