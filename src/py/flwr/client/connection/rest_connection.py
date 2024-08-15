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

import sys
from logging import DEBUG
from pathlib import Path
from typing import Any, Sequence, TypeVar, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import log
from flwr.common.constant import (
    GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY,
    GRPC_ADAPTER_METADATA_SHOULD_EXIT_KEY,
)
from flwr.common.grpc import create_channel
from flwr.common.version import package_version
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
from flwr.proto.grpcadapter_pb2 import MessageContainer  # pylint: disable=E0611
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterStub
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611

from .client_interceptor import AuthenticateClientInterceptor
from .fleet_api import FleetAPI
from .grpc_rere_connection import GrpcRereConnection, on_channel_state_change

import random
import sys
import threading
from contextlib import contextmanager
from copy import copy
from logging import ERROR, INFO, WARN
from typing import Callable, Iterator, Optional, Tuple, Type, TypeVar, Union

from cryptography.hazmat.primitives.asymmetric import ec
from google.protobuf.message import Message as GrpcMessage

from flwr.client.heartbeat import start_ping_loop
from flwr.client.message_handler.message_handler import validate_out_message
from flwr.client.message_handler.task_handler import get_task_ins, validate_task_ins
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.constant import (
    MISSING_EXTRA_REST,
    PING_BASE_MULTIPLIER,
    PING_CALL_TIMEOUT,
    PING_DEFAULT_INTERVAL,
    PING_RANDOM_RANGE,
)
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.serde import (
    message_from_taskins,
    message_to_taskres,
    user_config_from_proto,
)
from flwr.common.typing import Fab, Run
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

try:
    import requests
except ModuleNotFoundError:
    sys.exit(MISSING_EXTRA_REST)


PATH_CREATE_NODE: str = "api/v0/fleet/create-node"
PATH_DELETE_NODE: str = "api/v0/fleet/delete-node"
PATH_PULL_TASK_INS: str = "api/v0/fleet/pull-task-ins"
PATH_PUSH_TASK_RES: str = "api/v0/fleet/push-task-res"
PATH_PING: str = "api/v0/fleet/ping"
PATH_GET_RUN: str = "/api/v0/fleet/get-run"
PATH_GET_FAB: str = "/api/v0/fleet/get-fab"

T = TypeVar("T", bound=GrpcMessage)


class GrpcAdapterConnection(GrpcRereConnection):
    """Grpc-adapter connection based on GrpcRereConnection."""
    
    

    @property
    def api(self) -> FleetAPI:
        """The API proxy."""
        # NEVER SET VERIFY TO FALSE
        # Otherwise any server can fake its identity
        # Please refer to:
        # https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
        verify: Union[bool, str] = True
        if isinstance(self.root_certificates, str):
            verify = self.root_certificates
        elif isinstance(self.root_certificates, bytes):
            log(
                ERROR,
                "For the REST API, the root certificates "
                "must be provided as a string path to the client.",
            )
        if self.authentication_keys is not None:
            log(ERROR, "Client authentication is not supported for this transport type.")

        return RestFleetAPI(self.server_address, verify)


class RestFleetAPI(FleetAPI):
    """Adapter class to send and receive gRPC messages via the ``GrpcAdapterStub``.

    This class utilizes the ``GrpcAdapterStub`` to send and receive gRPC messages
    which are defined and used by the Fleet API, as defined in ``fleet.proto``.
    """

    def __init__(self, base_url: str, verify: bool | str) -> None:
        self.base_url = base_url
        self.verify = verify

    def _request(
        self, req: GrpcMessage, res_type: Type[T], api_path: str, timeout: float | None
    ) -> Optional[T]:
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
            return None
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                api_path,
            )
            return None
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                api_path,
            )
            return None

        # Deserialize ProtoBuf from bytes
        grpc_res = res_type()
        grpc_res.ParseFromString(res.content)
        return grpc_res

    def CreateNode(  # pylint: disable=C0103
        self, request: CreateNodeRequest, **kwargs: Any
    ) -> CreateNodeResponse:
        """."""
        timeout = kwargs.get("timeout", None)
        return self._request(request, CreateNodeResponse, PATH_CREATE_NODE, timeout)

    def DeleteNode(  # pylint: disable=C0103
        self, request: DeleteNodeRequest, **kwargs: Any
    ) -> DeleteNodeResponse:
        """."""
        timeout = kwargs.get("timeout", None)
        return self._request(request, DeleteNodeResponse, PATH_DELETE_NODE, timeout)

    def Ping(  # pylint: disable=C0103
        self, request: PingRequest, **kwargs: Any
    ) -> PingResponse:
        """."""
        timeout = kwargs.get("timeout", None)
        return self._request(request, PingResponse, PATH_PING, timeout)

    def PullTaskIns(  # pylint: disable=C0103
        self, request: PullTaskInsRequest, **kwargs: Any
    ) -> PullTaskInsResponse:
        """."""
        timeout = kwargs.get("timeout", None)
        return self._request(request, PullTaskInsResponse, PATH_PULL_TASK_INS, timeout)

    def PushTaskRes(  # pylint: disable=C0103
        self, request: PushTaskResRequest, **kwargs: Any
    ) -> PushTaskResResponse:
        """."""
        timeout = kwargs.get("timeout", None)
        return self._request(request, PushTaskResResponse, PATH_PUSH_TASK_RES, timeout)

    def GetRun(  # pylint: disable=C0103
        self, request: GetRunRequest, **kwargs: Any
    ) -> GetRunResponse:
        """."""
        timeout = kwargs.get("timeout", None)
        return self._request(request, GetRunResponse, PATH_GET_RUN, timeout)

    def GetFab(  # pylint: disable=C0103
        self, request: GetFabRequest, **kwargs: Any
    ) -> GetFabResponse:
        """."""
        timeout = kwargs.get("timeout", None)
        return self._request(request, GetFabResponse, PATH_GET_FAB, timeout)
