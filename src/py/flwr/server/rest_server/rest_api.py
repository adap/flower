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

from asgiref.sync import async_to_sync
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.datastructures import Headers

from flwr.common.logger import log
from flwr.proto.fleet_pb2 import (
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)

app = FastAPI()


@app.post("/api/v0/fleet/pull-task-ins", response_class=Response)
def pull_task_ins(request: Request) -> Response:
    """."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    pull_task_ins_request_bytes: bytes = async_to_sync(request.body)()  # type: ignore

    # Deserialize ProtoBuf
    pull_task_ins_request_proto = PullTaskInsRequest()
    pull_task_ins_request_proto.ParseFromString(pull_task_ins_request_bytes)

    # Print received message
    log(INFO, "POST - Receiving GetTaskRequest %s", pull_task_ins_request_proto)

    # TODO get pull_task_ins from state

    # Return serialized ProtoBuf
    pull_task_ins_response_proto = PullTaskInsResponse()
    pull_task_ins_response_bytes = pull_task_ins_response_proto.SerializeToString()
    log(INFO, "POST - Returning PullTaskInsResponse %s", pull_task_ins_response_proto)
    return Response(
        status_code=200,
        content=pull_task_ins_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


@app.post("/api/v0/fleet/push-task-res")
def push_task_res(request: Request) -> Response:  # Check if token is needed here
    """."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    push_task_res_request_bytes: bytes = async_to_sync(request.body)()  # type: ignore

    # Deserialize ProtoBuf
    push_task_res_request_proto = PushTaskResRequest()
    push_task_res_request_proto.ParseFromString(push_task_res_request_bytes)

    # Print received message
    log(INFO, "POST - Receiving PushTaskResRequest %", push_task_res_request_proto)

    # TODO get pull_task_ins from state

    # Return serialized ProtoBuf
    push_task_res_response_proto = PushTaskResResponse()
    push_task_res_response_bytes = push_task_res_response_proto.SerializeToString()
    log(INFO, "POST - Returning PushTaskResResponse %s", push_task_res_response_proto)
    return Response(
        status_code=200,
        content=push_task_res_response_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


def _check_headers(headers: Headers) -> None:
    """Check if expected headers are set."""
    if not "content-type" in headers:
        raise HTTPException(status_code=400, detail="Missing header `Content-Type`")
    if headers["content-type"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported `Content-Type`")
    if not "accept" in headers:
        raise HTTPException(status_code=400, detail="Missing header `Accept`")
    if headers["accept"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported `Accept`")
