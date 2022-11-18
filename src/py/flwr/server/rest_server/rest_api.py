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
from typing import Dict, Optional

from fastapi import FastAPI, Request, Response

from flwr.proto.fleet_pb2 import (
    CreateResultsRequest,
    CreateResultsResponse,
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


@app.get("/api/1.1/tasks/", response_class=Response)
def tasks(response: Response, client_id: Optional[int] = None):
    # Create a mock message. See mock_msgs.py for more details.
    message_being_sent = choice(list(mock_messages.keys()))
    tokenized_task = mock_messages[message_being_sent]

    # Return serialized ProtoBuf
    task_resp_msg = GetTasksResponse()
    task_resp_msg.tokenized_tasks.tokenized_tasks.append(tokenized_task)
    task_resp_bytes = task_resp_msg.SerializeToString()
    print(f"GET - Sending Task {message_being_sent}:")
    print(task_resp_msg)
    return Response(
        status_code=200,
        content=task_resp_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


@app.post("/api/1.1/result/")
async def result(request: Request):  # Check if token is needed here
    # This is required to get the request body as raw bytes
    create_results_req_msg_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    create_results_req_msg = CreateResultsRequest()
    create_results_req_msg.ParseFromString(create_results_req_msg_bytes)

    # Print received message
    print(f"POST - Receiving Result:")
    print(create_results_req_msg)

    # Create response
    create_results_resp_msg = CreateResultsResponse()
    create_results_resp_msg_bytes = create_results_resp_msg.SerializeToString()
    return Response(
        status_code=200,
        content=create_results_resp_msg_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


###############################################################################
# Client availability
###############################################################################


# @app.post("/api/1.1/client/register/{client_id}")
# def register(client_id: str, token: str):
#    pass


# @app.delete("/api/1.1/client/unregister/{client_id}")
# def unregister(client_id: str):
#    pass


# @app.put("/api/1.1/client/heartbeat/{client_id}")
# def client_heartbeat(client_id: str):
#    pass
