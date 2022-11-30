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


from random import choice
from typing import Dict

from asgiref.sync import async_to_sync
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.datastructures import Headers

from flwr.proto.fleet_pb2 import (
    CreateResultsRequest,
    CreateResultsResponse,
    GetTasksRequest,
    GetTasksResponse,
    TokenizedTask,
)
from flwr.server.rest_server.mock_msgs import gen_mock_messages

# tm = TaskManagerDB()
# cm = ClientManagerDB()

app = FastAPI()

###############################################################################
# Dummy messages
###############################################################################
mock_messages: Dict[str, TokenizedTask] = gen_mock_messages()


###############################################################################
# Tasks and results
###############################################################################


@app.post("/api/1.1/tasks", response_class=Response)
def tasks(request: Request) -> Response:
    """."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    get_tasks_req_msg_bytes: bytes = async_to_sync(request.body)()

    # Deserialize ProtoBuf
    get_tasks_req_msg = GetTasksRequest()
    get_tasks_req_msg.ParseFromString(get_tasks_req_msg_bytes)

    # Print received message
    print(f"POST - Receiving GetTaskRequest:")
    print(get_tasks_req_msg)

    # Create a mock message. See mock_msgs.py for more details.
    message_being_sent = choice(list(mock_messages.keys()))
    tokenized_task = mock_messages[message_being_sent]

    # Return serialized ProtoBuf
    task_resp_msg = GetTasksResponse()
    task_resp_msg.tokenized_tasks.tokenized_tasks.append(tokenized_task)
    task_resp_bytes = task_resp_msg.SerializeToString()
    print(f"POST - Sending GetTasksResponse {message_being_sent}:")
    print(task_resp_msg)
    return Response(
        status_code=200,
        content=task_resp_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


@app.post("/api/1.1/results")
def results(request: Request) -> Response:  # Check if token is needed here
    """."""
    _check_headers(request.headers)

    # Get the request body as raw bytes
    create_results_req_msg_bytes: bytes = async_to_sync(request.body)()

    # Deserialize ProtoBuf
    create_results_req_msg = CreateResultsRequest()
    create_results_req_msg.ParseFromString(create_results_req_msg_bytes)

    # Print received message
    print(f"POST - Receiving CreateResultsRequest:")
    print(create_results_req_msg)

    # Create response
    create_results_resp_msg = CreateResultsResponse()
    create_results_resp_msg_bytes = create_results_resp_msg.SerializeToString()
    print(f"POST - Sending CreateResultsResponse {create_results_resp_msg}:")
    return Response(
        status_code=200,
        content=create_results_resp_msg_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


# def _check_headers(headers: Dict[str, Union[str, bytes]]) -> None:
def _check_headers(headers: Headers) -> None:
    """Check if expected headers are set."""
    if not "content-type" in headers:
        raise HTTPException(status_code=400, detail="Missing header Content-Type")
    if headers["content-type"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported Content-Type")
    if not "accept" in headers:
        raise HTTPException(status_code=400, detail="Missing header Accept")
    if headers["accept"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported Accept")
