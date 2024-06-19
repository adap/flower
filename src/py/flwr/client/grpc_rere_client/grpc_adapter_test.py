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
"""Tests for the GrpcAdapter class."""


import inspect

from flwr.proto.fleet_pb2_grpc import FleetServicer

from .grpc_adapter import GrpcAdapter


def test_grpc_adapter_methods() -> None:
    """Test if GrpcAdapter implements all required methods."""
    # Prepare
    methods = {
        name for name, ref in inspect.getmembers(GrpcAdapter) if inspect.isfunction(ref)
    }
    expected_methods = {
        name
        for name, ref in inspect.getmembers(FleetServicer)
        if inspect.isfunction(ref)
    }

    # Assert
    assert expected_methods.issubset(methods)