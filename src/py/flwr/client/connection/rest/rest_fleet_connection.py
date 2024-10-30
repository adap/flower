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
"""Connection for a REST request-response channel to the SuperLink."""


from __future__ import annotations

import importlib.util
import sys
from logging import ERROR, WARN
from typing import Any, TypeVar

from google.protobuf.message import Message as GrpcMessage

from flwr.common import log
from flwr.common.constant import MISSING_EXTRA_REST
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

from ..fleet_api import FleetApi
from ..rere_fleet_connection import RereFleetConnection

if importlib.util.find_spec("requests"):
    import requests
    from requests.exceptions import HTTPError, InvalidHeader


PATH_CREATE_NODE: str = "api/v0/fleet/create-node"
PATH_DELETE_NODE: str = "api/v0/fleet/delete-node"
PATH_PULL_TASK_INS: str = "api/v0/fleet/pull-task-ins"
PATH_PUSH_TASK_RES: str = "api/v0/fleet/push-task-res"
PATH_PING: str = "api/v0/fleet/ping"
PATH_GET_RUN: str = "/api/v0/fleet/get-run"
PATH_GET_FAB: str = "/api/v0/fleet/get-fab"

T = TypeVar("T", bound=GrpcMessage)


class RestFleetConnection(RereFleetConnection):
    """Rest fleet connection based on RereFleetConnection."""

    _api: FleetApi | None = None

    @property
    def api(self) -> FleetApi:
        """The proxy providing low-level access to the Fleet API server."""
        if self._api is None:
            # Initialize the connection to the SuperLink Fleet API server

            # NEVER SET VERIFY TO FALSE
            # Otherwise any server can fake its identity
            # Please refer to:
            # https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
            verify: bool | str = True
            if isinstance(self.root_certificates, str):
                verify = self.root_certificates
            elif isinstance(self.root_certificates, bytes):
                log(
                    ERROR,
                    "For the REST API, the root certificates "
                    "must be provided as a string path to the client.",
                )
            if self.authentication_keys is not None:
                log(
                    ERROR,
                    "Client authentication is not supported for this transport type.",
                )

            self._api = RestFleetApi(self.server_address, verify)

        return self._api


class RestFleetApi(FleetApi):
    """Adapter class to send and receive gRPC messages via HTTP.

    This class utilizes the ``requests`` module to send and receive gRPC messages
    which are defined and used by the Fleet API, as defined in ``fleet.proto``.
    """

    def __init__(self, base_url: str, verify: bool | str) -> None:
        self.base_url = base_url
        self.verify = verify

        # Check the availability of the requests module
        if not importlib.util.find_spec("requests"):
            sys.exit(MISSING_EXTRA_REST)

    def _request(
        self, req: GrpcMessage, res_type: type[T], api_path: str, **kwargs: Any
    ) -> T:
        # Read kwargs
        timeout = kwargs.pop("timeout", None)
        for key in kwargs:  # Unsupported arguments
            log(WARN, "Unsupported argument: `%s`", key)

        # Serialize the request
        req_bytes = req.SerializeToString()

        # Send the request
        def post() -> requests.Response:
            return requests.post(
                f"{self.base_url}/{api_path}",
                data=req_bytes,
                headers={
                    "Accept": "application/protobuf",
                    "Content-Type": "application/protobuf",
                },
                verify=self.verify,
                timeout=timeout,
            )

        res = post()

        # Check status code and headers
        if res.status_code != 200:
            # pylint: disable-next=possibly-used-before-assignment
            raise HTTPError(f"Unexpected status code: {res.status_code}")
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                api_path,
            )
            # pylint: disable-next=possibly-used-before-assignment
            raise InvalidHeader("Missing `Content-Type` header in the response")
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                api_path,
            )
            # pylint: disable-next=possibly-used-before-assignment
            raise InvalidHeader(
                "`Content-Type` header has wrong value, expected `application/protobuf`"
            )

        # Deserialize ProtoBuf from bytes
        grpc_res = res_type.FromString(res.content)
        return grpc_res

    def CreateNode(  # pylint: disable=C0103
        self, request: CreateNodeRequest, **kwargs: Any
    ) -> CreateNodeResponse:
        """."""
        return self._request(request, CreateNodeResponse, PATH_CREATE_NODE, **kwargs)

    def DeleteNode(  # pylint: disable=C0103
        self, request: DeleteNodeRequest, **kwargs: Any
    ) -> DeleteNodeResponse:
        """."""
        return self._request(request, DeleteNodeResponse, PATH_DELETE_NODE, **kwargs)

    def Ping(  # pylint: disable=C0103
        self, request: PingRequest, **kwargs: Any
    ) -> PingResponse:
        """."""
        return self._request(request, PingResponse, PATH_PING, **kwargs)

    def PullTaskIns(  # pylint: disable=C0103
        self, request: PullTaskInsRequest, **kwargs: Any
    ) -> PullTaskInsResponse:
        """."""
        return self._request(request, PullTaskInsResponse, PATH_PULL_TASK_INS, **kwargs)

    def PushTaskRes(  # pylint: disable=C0103
        self, request: PushTaskResRequest, **kwargs: Any
    ) -> PushTaskResResponse:
        """."""
        return self._request(request, PushTaskResResponse, PATH_PUSH_TASK_RES, **kwargs)

    def GetRun(  # pylint: disable=C0103
        self, request: GetRunRequest, **kwargs: Any
    ) -> GetRunResponse:
        """."""
        return self._request(request, GetRunResponse, PATH_GET_RUN, **kwargs)

    def GetFab(  # pylint: disable=C0103
        self, request: GetFabRequest, **kwargs: Any
    ) -> GetFabResponse:
        """."""
        return self._request(request, GetFabResponse, PATH_GET_FAB, **kwargs)
