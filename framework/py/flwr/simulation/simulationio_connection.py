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
"""Flower SimulationIo connection."""


from logging import DEBUG, WARNING
from typing import Optional, cast

import grpc

from flwr.common.constant import SIMULATIONIO_API_DEFAULT_CLIENT_ADDRESS
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.proto.simulationio_pb2_grpc import SimulationIoStub  # pylint: disable=E0611


class SimulationIoConnection:
    """`SimulationIoConnection` provides an interface to the SimulationIo API.

    Parameters
    ----------
    simulationio_service_address : str (default: "[::]:9094")
        The address (URL, IPv6, IPv4) of the SuperLink SimulationIo API service.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        simulationio_service_address: str = SIMULATIONIO_API_DEFAULT_CLIENT_ADDRESS,
        root_certificates: Optional[bytes] = None,
    ) -> None:
        self._addr = simulationio_service_address
        self._cert = root_certificates
        self._grpc_stub: Optional[SimulationIoStub] = None
        self._channel: Optional[grpc.Channel] = None
        self._retry_invoker = _make_simple_grpc_retry_invoker()

    @property
    def _is_connected(self) -> bool:
        """Check if connected to the SimulationIo API server."""
        return self._channel is not None

    @property
    def _stub(self) -> SimulationIoStub:
        """SimulationIo stub."""
        if not self._is_connected:
            self._connect()
        return cast(SimulationIoStub, self._grpc_stub)

    def _connect(self) -> None:
        """Connect to the SimulationIo API."""
        if self._is_connected:
            log(WARNING, "Already connected")
            return
        self._channel = create_channel(
            server_address=self._addr,
            insecure=(self._cert is None),
            root_certificates=self._cert,
        )
        self._channel.subscribe(on_channel_state_change)
        self._grpc_stub = SimulationIoStub(self._channel)
        _wrap_stub(self._grpc_stub, self._retry_invoker)
        log(DEBUG, "[SimulationIO] Connected to %s", self._addr)

    def _disconnect(self) -> None:
        """Disconnect from the SimulationIo API."""
        if not self._is_connected:
            log(DEBUG, "Already disconnected")
            return
        channel: grpc.Channel = self._channel
        self._channel = None
        self._grpc_stub = None
        channel.close()
        log(DEBUG, "[SimulationIO] Disconnected")
