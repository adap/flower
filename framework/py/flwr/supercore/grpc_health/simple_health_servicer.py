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
"""Simple gRPC health servicers."""


import grpc

# pylint: disable=E0611
from grpc_health.v1.health_pb2 import HealthCheckRequest, HealthCheckResponse
from grpc_health.v1.health_pb2_grpc import HealthServicer

# pylint: enable=E0611


class SimpleHealthServicer(HealthServicer):  # type: ignore
    """A simple gRPC health servicer that always returns SERVING."""

    def Check(
        self, request: HealthCheckRequest, context: grpc.ServicerContext
    ) -> HealthCheckResponse:
        """Return a HealthCheckResponse with SERVING status."""
        return HealthCheckResponse(status=HealthCheckResponse.SERVING)

    def Watch(self, request: HealthCheckRequest, context: grpc.ServicerContext) -> None:
        """Watch the health status (not implemented)."""
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Watch is not implemented")
