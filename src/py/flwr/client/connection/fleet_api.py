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
"""Fleet API definition for the grpc-rere transport layer."""


from abc import ABC, abstractmethod
from typing import Any

from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611


class FleetApi(ABC):
    """Fleet API that provides low-level access to Fleet API server."""

    @abstractmethod
    def Ping(  # pylint: disable=C0103
        self, request: PingRequest, **kwargs: Any
    ) -> PingResponse:
        """Fleet.Ping."""

    @abstractmethod
    def CreateNode(  # pylint: disable=C0103
        self, request: CreateNodeRequest, **kwargs: Any
    ) -> CreateNodeResponse:
        """Fleet.CreateNode."""

    @abstractmethod
    def DeleteNode(  # pylint: disable=C0103
        self, request: DeleteNodeRequest, **kwargs: Any
    ) -> DeleteNodeResponse:
        """Fleet.DeleteNode."""

    @abstractmethod
    def PullTaskIns(  # pylint: disable=C0103
        self, request: PullTaskInsRequest, **kwargs: Any
    ) -> PullTaskInsResponse:
        """Fleet.PullTaskIns."""

    @abstractmethod
    def PushTaskRes(  # pylint: disable=C0103
        self, request: PushTaskResRequest, **kwargs: Any
    ) -> PushTaskResResponse:
        """Fleet.PushTaskRes."""

    @abstractmethod
    def GetRun(  # pylint: disable=C0103
        self, request: GetRunRequest, **kwargs: Any
    ) -> GetRunResponse:
        """Fleet.GetRun."""

    @abstractmethod
    def GetFab(  # pylint: disable=C0103
        self, request: GetFabRequest, **kwargs: Any
    ) -> GetFabResponse:
        """Fleet.GetFab."""
