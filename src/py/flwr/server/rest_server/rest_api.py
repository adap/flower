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

from fastapi import FastAPI, Request, Response, status

from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.rest_server.rest_client_proxy import RestClientProxy
from flwr.server.rest_server.rest_server import State

app = FastAPI()


###############################################################################
# Client availability
###############################################################################


@app.post("/client/{client_id}")
def client_here(client_id: str):
    print(f"client-HERE {client_id}")

    # Get the current ClientManager
    state = State.instance()
    cm = state.get_client_manager()

    # TODO check if there is a RestClientProxy for that ID, and, if so, return error

    # Register new ClientProxy
    cp = RestClientProxy(cid=client_id)
    cm.register(cp)


@app.delete("/client/{client_id}")
def client_away(client_id: str):
    print(f"client-AWAY {client_id}")

    # Get the current ClientManager
    state = State.instance()
    cm = state.get_client_manager()

    # TODO check if there is a RestClientProxy for that ID, and, if not, return error

    # Unregister new ClientProxy
    cp = RestClientProxy(cid=client_id)
    cm.unregister(cp)


@app.put("/client/{client_id}")
def client_heartbeat(client_id: str):
    print(f"client-HEARTBEAT {client_id}")
    # TODO handle heartbeat and expiration everywhere
    # TODO return heartbeat frequency / timeout to client?


###############################################################################
# Tasks and results
###############################################################################


@app.get("/ins/{client_id}", response_class=Response)
def ins(client_id: str, response: Response):
    print(f"client-INS {client_id}")

    # Get the current TaskManager
    state = State.instance()
    tm = state.get_task_manager()

    # See if TaskManager has a task for this client
    client_msg: Optional[ServerMessage] = tm.get_task(cid=client_id)
    if client_msg is None:
        print(f"Client {client_id} has nothing to do")
        response.status_code = status.HTTP_418_IM_A_TEAPOT  # TODO
        return response

    print(f"Client {client_id} has work to do")

    # Return serialized ProtoBuf
    client_msg_bytes = client_msg.SerializeToString()
    return Response(
        status_code=200,
        content=client_msg_bytes,
        headers={"Content-Type": "application/protobuf"},
    )


@app.post("/res/{client_id}")
async def res(client_id: str, request: Request):
    print(f"client-RES {client_id}")

    # This is required to get the request body as raw bytes
    client_msg_bytes: bytes = await request.body()

    # Deserialize ProtoBuf
    client_msg = ClientMessage()
    client_msg.ParseFromString(client_msg_bytes)

    # Get current TaskManager
    state = State.instance()
    tm = state.get_task_manager()

    # Set task result
    tm.set_result(cid=client_id, client_message=client_msg)
