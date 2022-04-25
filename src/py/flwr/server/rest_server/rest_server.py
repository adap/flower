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
"""Create and start a REST server."""


import os
import threading

import uvicorn

from flwr.common import GRPC_MAX_MESSAGE_LENGTH


def start_rest_server(  # pylint: disable=too-many-arguments
    server_address: str = "0.0.0.0:8080",
) -> None:
    """Create REST server."""
    print(f"start_rest_server, PID:", os.getpid())

    # Start REST server on a different thread
    rest_server_thread = threading.Thread(target=rest_server, args=[server_address])
    rest_server_thread.start()
    return rest_server_thread


def rest_server(server_address: str):
    """Start a FastAPI server and block until it is done."""

    # Split server address in host and port
    host, port = server_address.split(":")  # TODO support IPv6

    # Start FastAPI server
    uvicorn.run(
        "flwr.server.rest_server.rest_api:app",
        host=host,
        port=int(port),
        log_level="info",
        debug=False,
    )
