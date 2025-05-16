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

import tempfile
import unittest
from typing import Union

import grpc
import numpy as np
from parameterized import parameterized

from flwr.common import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common.constant import (
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
)
from flwr.proto.fleet_pb2_grpc import FleetStub
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.server.app import _run_fleet_api_grpc_rere
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc

from .inflatable_grpc_utils import push_object_to_servicer

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
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)  # Ensures cleanup after test

        state_factory = LinkStateFactory(":flwr-in-memory-state:")
        self.state = state_factory.state()
        ffs_factory = FfsFactory(self.temp_dir.name)
        self.ffs = ffs_factory.ffs()

        # ServerAppIo endpoints
        self._server_serverappio: grpc.Server = run_serverappio_api_grpc(
            SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
            state_factory,
            ffs_factory,
            None,
        )
        self._channel_serverappio = grpc.insecure_channel("localhost:9091")
        self._push_object_serverappio = self._channel_serverappio.unary_unary(
            "/flwr.proto.ServerAppIo/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )

        # Fleet endpoints
        self._server_fleet: grpc.Server = _run_fleet_api_grpc_rere(
            FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
            state_factory,
            ffs_factory,
            None,
            None,
        )
        self._channel_fleet = grpc.insecure_channel("localhost:9092")
        self._push_object_fleet = self._channel_fleet.unary_unary(
            "/flwr.proto.Fleet/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server_serverappio.stop(None)
        self._server_fleet.stop(None)

    def get_serverappio_stub(self) -> ServerAppIoStub:
        """Get ServerAppIo stub."""
        stub = ServerAppIoStub(self._channel_serverappio)
        stub.PushObject = self._push_object_serverappio
        return stub

    def get_fleet_stub(self) -> FleetStub:
        """Get ServerAppIo stub."""
        stub = FleetStub(self._channel_fleet)
        stub.PushObject = self._push_object_fleet
        return stub

    @parameterized.expand(
        [
            (records, expected, stub_type)
            for records, expected in base_cases
            for stub_type in [ServerAppIoStub, FleetStub]
        ]
    )  # type: ignore
    def test_push_object_with_helper_function(
        self,
        records: dict[str, Union[ArrayRecord, ConfigRecord, MetricRecord]],
        expected_obj_count: int,
        stub_type: type[Union[ServerAppIoStub, FleetStub]],
    ) -> None:
        """Use helper function to push an object recursively."""
        # Prepare
        obj = RecordDict(records)
        # Construct a stub and use mocked push_object
        stub: Union[ServerAppIoStub, FleetStub]
        if stub_type == ServerAppIoStub:
            stub = self.get_serverappio_stub()
        elif stub_type == FleetStub:
            stub = self.get_fleet_stub()
        else:
            raise NotImplementedError()

        # Execute
        pushed_object_ids = push_object_to_servicer(obj, stub)

        # Assert
        # Expected number of objects were pushed
        assert (
            len(pushed_object_ids) == expected_obj_count + 1
        )  # +1 due to the RecordDict itself
