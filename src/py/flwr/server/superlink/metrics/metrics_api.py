# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Experimental metrics API server."""

import os
import sys
from datetime import datetime

import psutil

from flwr.common.constant import MISSING_EXTRA_REST
from flwr.server.superlink.state import State

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route
except ModuleNotFoundError:
    sys.exit(MISSING_EXTRA_REST)


def getstarttime(pid=None):
    if not pid:
        pid = os.getpid()
    p = psutil.Process(pid)
    return datetime.fromtimestamp(p.create_time())


def getuptime(pid=None):
    return int((datetime.now() - getstarttime(pid)).total_seconds())


async def get_metrics(request: Request) -> Response:
    """Return metrics."""
    # Get state from app
    state: State = app.state.STATE_FACTORY.state()

    metrics = {
        "uptime": getuptime(),
        "num_task_ins": state.num_task_ins(),
        "num_task_res": state.num_task_res(),
    }

    # Return serialized ProtoBuf
    return JSONResponse(
        status_code=200,
        content=metrics,
        headers={"Content-Type": "application/json"},
    )


routes = [
    Route("/api/v0/metrics", get_metrics, methods=["GET"]),
]

app: Starlette = Starlette(
    debug=False,
    routes=routes,
)
