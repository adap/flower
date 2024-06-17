# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


import sys

from flwr.common.constant import MISSING_EXTRA_REST
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    DeleteNodeRequest,
    PingRequest,
    PullTaskInsRequest,
    PushTaskResRequest,
)
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.server.superlink.fleet.message_handler import message_handler
from flwr.server.superlink.state import State

try:
    from starlette.applications import Starlette
    from starlette.datastructures import Headers
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route
except ModuleNotFoundError:
    sys.exit(MISSING_EXTRA_REST)


async def create_node(request: Request) -> Response:
    """Create Node."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    create_node_request_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    create_node_request_proto = CreateNodeRequest()
    create_node_request_proto.ParseFromString(create_node_request_bytes)

    # Get state from app
    state: State = app.state.STATE_FACTORY.state()

    # Handle message
    create_node_response_proto = message_handler.create_node(
        request=create_node_request_proto, state=state
    )

    # Return serialized ProtoBuf
    create_node_response_bytes = create_node_response_proto.SerializeToString()
    return Response(
        status_code=200,
        content=create_node_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


async def delete_node(request: Request) -> Response:
    """Delete Node Id."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    delete_node_request_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    delete_node_request_proto = DeleteNodeRequest()
    delete_node_request_proto.ParseFromString(delete_node_request_bytes)

    # Get state from app
    state: State = app.state.STATE_FACTORY.state()

    # Handle message
    delete_node_response_proto = message_handler.delete_node(
        request=delete_node_request_proto, state=state
    )

    # Return serialized ProtoBuf
    delete_node_response_bytes = delete_node_response_proto.SerializeToString()
    return Response(
        status_code=200,
        content=delete_node_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


async def pull_task_ins(request: Request) -> Response:
    """Pull TaskIns."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    pull_task_ins_request_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    pull_task_ins_request_proto = PullTaskInsRequest()
    pull_task_ins_request_proto.ParseFromString(pull_task_ins_request_bytes)

    # Get state from app
    state: State = app.state.STATE_FACTORY.state()

    # Handle message
    pull_task_ins_response_proto = message_handler.pull_task_ins(
        request=pull_task_ins_request_proto,
        state=state,
    )

    # Return serialized ProtoBuf
    pull_task_ins_response_bytes = pull_task_ins_response_proto.SerializeToString()
    return Response(
        status_code=200,
        content=pull_task_ins_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


async def push_task_res(request: Request) -> Response:  # Check if token is needed here
    """Push TaskRes."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    push_task_res_request_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    push_task_res_request_proto = PushTaskResRequest()
    push_task_res_request_proto.ParseFromString(push_task_res_request_bytes)

    # Get state from app
    state: State = app.state.STATE_FACTORY.state()

    # Handle message
    push_task_res_response_proto = message_handler.push_task_res(
        request=push_task_res_request_proto,
        state=state,
    )

    # Return serialized ProtoBuf
    push_task_res_response_bytes = push_task_res_response_proto.SerializeToString()
    return Response(
        status_code=200,
        content=push_task_res_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


async def ping(request: Request) -> Response:
    """Ping."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    ping_request_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    ping_request_proto = PingRequest()
    ping_request_proto.ParseFromString(ping_request_bytes)

    # Get state from app
    state: State = app.state.STATE_FACTORY.state()

    # Handle message
    ping_response_proto = message_handler.ping(request=ping_request_proto, state=state)

    # Return serialized ProtoBuf
    ping_response_bytes = ping_response_proto.SerializeToString()
    return Response(
        status_code=200,
        content=ping_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


async def get_run(request: Request) -> Response:
    """GetRun."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    get_run_request_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    get_run_request_proto = GetRunRequest()
    get_run_request_proto.ParseFromString(get_run_request_bytes)

    # Get state from app
    state: State = app.state.STATE_FACTORY.state()

    # Handle message
    get_run_response_proto = message_handler.get_run(
        request=get_run_request_proto, state=state
    )

    # Return serialized ProtoBuf
    get_run_response_bytes = get_run_response_proto.SerializeToString()
    return Response(
        status_code=200,
        content=get_run_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


routes = [
    Route("/api/v0/fleet/create-node", create_node, methods=["POST"]),
    Route("/api/v0/fleet/delete-node", delete_node, methods=["POST"]),
    Route("/api/v0/fleet/pull-task-ins", pull_task_ins, methods=["POST"]),
    Route("/api/v0/fleet/push-task-res", push_task_res, methods=["POST"]),
    Route("/api/v0/fleet/ping", ping, methods=["POST"]),
    Route("/api/v0/fleet/get-run", get_run, methods=["POST"]),
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
