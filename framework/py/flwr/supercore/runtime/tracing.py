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
"""Optional tracing interceptor owned by Runtime API callers."""

import httpx

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskResponse,
    PullTaskInputResponse,
)
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext
from flwr.supercore.tracing import current_traceparent, validate_traceparent


class RuntimeTracingInterceptor:
    """Propagate headers and retain response carriers by task, not by last call."""

    def __init__(self) -> None:
        self._task_traceparents: dict[int, str] = {}

    def intercept(
        self, context: ProtobufRequestContext, call_next: ProtobufCall
    ) -> httpx.Response:
        """Keep telemetry failures separate from transport and application errors."""
        try:
            if carrier := current_traceparent():
                context.request.headers["traceparent"] = carrier
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        response = call_next(context)
        try:
            if response.is_success:
                # Inspect only task-acquisition replies. The transport still owns
                # normal protobuf decoding and reports malformed payloads itself.
                if context.rpc_method == "/flwr.proto.Runtime/AcquireTask":
                    acquired = AcquireTaskResponse.FromString(response.content)
                    if not acquired.HasField("task"):
                        return response
                    task_id = acquired.task.task_id
                elif context.rpc_method == "/flwr.proto.Runtime/PullTaskInput":
                    task_id = PullTaskInputResponse.FromString(response.content).task_id
                else:
                    return response
                self._task_traceparents.pop(task_id, None)
                carrier = validate_traceparent(response.headers.get("traceparent"))
                if carrier:
                    self._task_traceparents[task_id] = carrier
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return response

    def take_task_traceparent(self, task_id: int) -> str:
        """Consume metadata for one task after a successful Runtime call."""
        return self._task_traceparents.pop(task_id, "")
