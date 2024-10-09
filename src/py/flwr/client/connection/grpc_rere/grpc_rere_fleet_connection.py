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
"""Connection for a gRPC request-response channel to the SuperLink."""


from __future__ import annotations

from collections.abc import Sequence
from logging import DEBUG
from pathlib import Path
from typing import cast

import grpc

from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.proto.fleet_pb2_grpc import FleetStub  # pylint: disable=E0611

from ..fleet_api import FleetApi
from ..rere_fleet_connection import RereFleetConnection
from .client_interceptor import AuthenticateClientInterceptor


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


class GrpcRereFleetConnection(RereFleetConnection):
    """Grpc-rere fleet connection based on RereFleetConnection."""

    _api: FleetApi | None = None

    @property
    def api(self) -> FleetApi:
        """The proxy providing low-level access to the Fleet API server."""
        if self._api is None:
            # Initialize the connection to the SuperLink Fleet API server
            if isinstance(self.root_certificates, str):
                root_cert: bytes | None = Path(self.root_certificates).read_bytes()
            else:
                root_cert = self.root_certificates
            interceptors: Sequence[grpc.UnaryUnaryClientInterceptor] | None = None
            if self.authentication_keys is not None:
                interceptors = AuthenticateClientInterceptor(*self.authentication_keys)

            self.channel = create_channel(
                server_address=self.server_address,
                insecure=self.insecure,
                root_certificates=root_cert,
                max_message_length=self.max_message_length,
                interceptors=interceptors,
            )
            self.channel.subscribe(on_channel_state_change)
            self._api = cast(FleetApi, FleetStub(self.channel))
        return self._api
