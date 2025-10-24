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
"""Experimental REST API server."""


from __future__ import annotations

from collections.abc import Awaitable
from typing import Callable, TypeVar, cast

from google.protobuf.message import Message as GrpcMessage

from flwr.common.exit import ExitCode, flwr_exit
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.superlink.fleet.message_handler import message_handler
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.ffs import Ffs, FfsFactory
from flwr.supercore.object_store import ObjectStore, ObjectStoreFactory

try:
    from starlette.applications import Starlette
    from starlette.datastructures import Headers
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route
except ModuleNotFoundError:
    flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)


GrpcRequest = TypeVar("GrpcRequest", bound=GrpcMessage)
GrpcResponse = TypeVar("GrpcResponse", bound=GrpcMessage)

GrpcAsyncFunction = Callable[[GrpcRequest], Awaitable[GrpcResponse]]
RestEndPoint = Callable[[Request], Awaitable[Response]]


def rest_request_response(
    grpc_request_type: type[GrpcRequest],
) -> Callable[[GrpcAsyncFunction[GrpcRequest, GrpcResponse]], RestEndPoint]:
    """Convert an async gRPC-based function into a RESTful HTTP endpoint."""

    def decorator(func: GrpcAsyncFunction[GrpcRequest, GrpcResponse]) -> RestEndPoint:
        async def wrapper(request: Request) -> Response:
            _check_headers(request.headers)

            # Get the request body as raw bytes
            grpc_req_bytes: bytes = await request.body()

            # Deserialize ProtoBuf
            grpc_req = grpc_request_type.FromString(grpc_req_bytes)
            grpc_res = await func(grpc_req)
            return Response(
                status_code=200,
                content=grpc_res.SerializeToString(),
                headers={"Content-Type": "application/protobuf"},
            )

        return wrapper

    return decorator


@rest_request_response(CreateNodeRequest)
async def create_node(request: CreateNodeRequest) -> CreateNodeResponse:
    """Create Node."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.create_node(request=request, state=state)


@rest_request_response(DeleteNodeRequest)
async def delete_node(request: DeleteNodeRequest) -> DeleteNodeResponse:
    """Delete Node Id."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.delete_node(request=request, state=state)


@rest_request_response(PullMessagesRequest)
async def pull_message(request: PullMessagesRequest) -> PullMessagesResponse:
    """Pull PullMessages."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.pull_messages(request=request, state=state, store=store)


@rest_request_response(PushMessagesRequest)
async def push_message(request: PushMessagesRequest) -> PushMessagesResponse:
    """Pull PushMessages."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.push_messages(request=request, state=state, store=store)


@rest_request_response(PullObjectRequest)
async def pull_object(request: PullObjectRequest) -> PullObjectResponse:
    """Pull PullObject."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.pull_object(request=request, state=state, store=store)


@rest_request_response(PushObjectRequest)
async def push_object(request: PushObjectRequest) -> PushObjectResponse:
    """Pull PushObject."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.push_object(request=request, state=state, store=store)


@rest_request_response(SendNodeHeartbeatRequest)
async def send_node_heartbeat(
    request: SendNodeHeartbeatRequest,
) -> SendNodeHeartbeatResponse:
    """Send node heartbeat."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.send_node_heartbeat(request=request, state=state)


@rest_request_response(GetRunRequest)
async def get_run(request: GetRunRequest) -> GetRunResponse:
    """GetRun."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.get_run(request=request, state=state, store=store)


@rest_request_response(GetFabRequest)
async def get_fab(request: GetFabRequest) -> GetFabResponse:
    """GetRun."""
    # Get ffs from app
    ffs: Ffs = cast(FfsFactory, app.state.FFS_FACTORY).ffs()

    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.get_fab(request=request, ffs=ffs, state=state, store=store)


@rest_request_response(ConfirmMessageReceivedRequest)
async def confirm_message_received(
    request: ConfirmMessageReceivedRequest,
) -> ConfirmMessageReceivedResponse:
    """Confirm message received."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.confirm_message_received(
        request=request, state=state, store=store
    )


routes = [
    Route("/api/v0/fleet/create-node", create_node, methods=["POST"]),
    Route("/api/v0/fleet/delete-node", delete_node, methods=["POST"]),
    Route("/api/v0/fleet/pull-messages", pull_message, methods=["POST"]),
    Route("/api/v0/fleet/push-messages", push_message, methods=["POST"]),
    Route("/api/v0/fleet/pull-object", pull_object, methods=["POST"]),
    Route("/api/v0/fleet/push-object", push_object, methods=["POST"]),
    Route("/api/v0/fleet/send-node-heartbeat", send_node_heartbeat, methods=["POST"]),
    Route("/api/v0/fleet/get-run", get_run, methods=["POST"]),
    Route("/api/v0/fleet/get-fab", get_fab, methods=["POST"]),
    Route(
        "/api/v0/fleet/confirm-message-received",
        confirm_message_received,
        methods=["POST"],
    ),
]

app: Starlette = Starlette(
    debug=False,
    routes=routes,
)


def _check_headers(headers: Headers) -> None:
    """Check if expected headers are set."""
    if "content-type" not in headers:
        raise HTTPException(status_code=400, detail="Missing header `Content-Type`")
    if headers["content-type"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported `Content-Type`")
    if "accept" not in headers:
        raise HTTPException(status_code=400, detail="Missing header `Accept`")
    if headers["accept"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported `Accept`")
