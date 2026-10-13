# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Best-effort tracing integration for the Runtime HTTP API."""

import os
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskResponse,
    PullTaskInputResponse,
)
from flwr.supercore.dependencies.runtime import (
    RuntimeStateDependency,
    TaskDependency,
    get_runtime_state,
)
from flwr.supercore.tracing import load_task_context, trace_span, validate_traceparent


class RuntimeTracingMiddleware(BaseHTTPMiddleware):
    """Attach task context inside the protobuf translation middleware."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Return the business response even when optional metadata fails."""
        response = await call_next(request)
        if os.getenv("FLWR_TRACING_ENABLED") != "1" or response.status_code >= 400:
            return response
        try:
            result = getattr(request.state, "protobuf_response", None)
            if isinstance(result, AcquireTaskResponse) and result.HasField("task"):
                task_id = result.task.task_id
            elif isinstance(result, PullTaskInputResponse):
                task_id = result.task_id
            else:
                return response

            def load_context() -> str:
                # Create/use synchronous state in the worker thread, not the loop.
                return load_task_context(get_runtime_state(request), task_id)

            if carrier := await run_in_threadpool(load_context):
                response.headers["traceparent"] = carrier
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return response


async def trace_task_completion(
    request: Request, state: RuntimeStateDependency, task: TaskDependency
) -> AsyncIterator[None]:
    """Scope completion tracing to the authenticated handler."""
    try:
        carrier = validate_traceparent(request.headers.get("traceparent"))
        if not carrier and os.getenv("FLWR_TRACING_ENABLED") == "1":
            carrier = await run_in_threadpool(load_task_context, state, task.task_id)
    except Exception:  # pylint: disable=broad-exception-caught
        carrier = ""
    with trace_span(
        "task.complete",
        traceparent=carrier,
        attributes={
            "flwr.task_id": str(task.task_id),
            "flwr.run_id": str(task.run_id),
        },
    ):
        yield


TaskCompletionTracingDependency = Annotated[
    None, Depends(trace_task_completion, scope="function")
]
