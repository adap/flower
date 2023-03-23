# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""REST API server."""


from logging import INFO
from typing import List, Optional
from uuid import UUID

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from starlette.datastructures import Headers
except ImportError as missing_dep:
    raise ImportError(
        "To use the REST API you must install the "
        "extra dependencies by running `pip install flwr['rest']`."
    ) from missing_dep

from flwr.common.logger import log
from flwr.proto.fleet_pb2 import (
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
    Reconnect,
)
from flwr.proto.task_pb2 import TaskIns, TaskRes
from flwr.server.state import State

app: FastAPI = FastAPI()


@app.post("/api/v0/fleet/pull-task-ins", response_class=Response)  # type: ignore
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

    # Retrieve TaskIns from State
    node = pull_task_ins_request_proto.node  # pylint: disable=no-member
    node_id: Optional[int] = None if node.anonymous else node.node_id
    task_ins_list: List[TaskIns] = state.get_task_ins(node_id=node_id, limit=1)
    pull_task_ins_response_proto = PullTaskInsResponse(task_ins_list=task_ins_list)

    # Return serialized ProtoBuf
    pull_task_ins_response_bytes = pull_task_ins_response_proto.SerializeToString()

    log(
        INFO,
        "POST - Returning PullTaskInsResponse %s",
        [ins.task_id for ins in task_ins_list],
    )
    return Response(
        status_code=200,
        content=pull_task_ins_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


@app.post("/api/v0/fleet/push-task-res", response_class=Response)  # type: ignore
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

    # Store TaskRes in State

    # pylint: disable=no-member
    task_res: TaskRes = push_task_res_request_proto.task_res_list[0]
    # pylint: enable=no-member

    task_id: Optional[UUID] = state.store_task_res(task_res=task_res)

    # Build response
    push_task_res_response_proto = PushTaskResResponse(
        reconnect=Reconnect(reconnect=5),
        results={str(task_id): 0},
    )

    # Return serialized ProtoBuf
    push_task_res_response_bytes = push_task_res_response_proto.SerializeToString()

    log(
        INFO,
        "POST - Returning PushTaskResResponse %s",
        push_task_res_response_proto,
    )
    return Response(
        status_code=200,
        content=push_task_res_response_bytes,
        headers={"Content-Type": "application/protobuf"},
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
