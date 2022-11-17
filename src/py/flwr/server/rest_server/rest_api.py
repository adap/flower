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
from random import random
from typing import Optional

from fastapi import FastAPI, Request, Response
from numpy import array

from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.serde import parameters_to_proto
from flwr.proto.fleet_pb2 import (
    CreateResultsRequest,
    CreateResultsResponse,
    GetTasksResponse,
    TokenizedTask,
)

# tm = TaskManagerDB()
# cm = ClientManagerDB()

app = FastAPI()

###############################################################################
# Tasks and results
###############################################################################


@app.get("/api/1.1/tasks/", response_class=Response)
def tasks(client_id: Optional[int], response: Response):
    # task_resp_msg = tm.get_tasks(client_id)
    # Create mock response and fill with something
    tokenized_task = TokenizedTask()
    tokenized_task.token = "abc"
    tokenized_task.task.task_id = 42
    if random() < 0.3:  # Tell client to check back 30% of the time
        tokenized_task.task.legacy_server_message.reconnect_ins.seconds = 5
    else:  # Send a fit_ins, missing config
        flwr_parameters = ndarrays_to_parameters([array([1, 2, 3]), array([4, 5, 6])])
        proto_parameters = parameters_to_proto(flwr_parameters)
        tokenized_task.task.legacy_server_message.fit_ins.parameters.CopyFrom(
            proto_parameters
        )

    task_resp_msg = GetTasksResponse()
    task_resp_msg.tokenized_tasks.tokenized_tasks.append(tokenized_task)

    # Return serialized ProtoBuf
    task_resp_bytes = task_resp_msg.SerializeToString()
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
