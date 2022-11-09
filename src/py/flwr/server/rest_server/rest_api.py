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


from typing import Optional

from fastapi import FastAPI, Request, Response

from flwr.proto.fleet_pb2 import (
    CreateResultsRequest,
    CreateResultsResponse,
    GetTasksResponse,
)

# tm = TaskManagerDB()
# cm = ClientManagerDB()

app = FastAPI()

###############################################################################
# Tasks and results
###############################################################################


@app.get("/api/1.1/tasks/", response_class=Response)
def tasks(client_id: Optional[int], response: Response):
    # Create mock response and fill with something
    task_resp_msg = GetTasksResponse()

    # Return serialized ProtoBuf
    task_resp_bytes = task_resp_msg.SerializeToString()
    return Response(
        status_code=200,
        content=task_resp_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


@app.post("/api/1.1/results")
async def results(client_id: str, request: Request):
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
