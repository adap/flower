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
"""Tests for the Runtime HTTP client."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from importlib import import_module
from threading import Barrier
from typing import Annotated
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    AcquireTaskResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.interceptors import RuntimeTokenHttpInterceptor
from flwr.supercore.protobuf.client import ProtobufClient
from flwr.supercore.protobuf.routing import ProtobufRoute
from flwr.supercore.protobuf.translation import (
    ProtobufTranslationMiddleware,
    get_protobuf_request,
)
from flwr.supercore.routers.runtime.router import acquire_task, pull_task_input
from flwr.supercore.runtime import RuntimeHttpClient

_UNARY_UNARY_PATHS = (
    "acquire-task",
    "send-task-heartbeat",
    "pull-task-input",
    "push-task-output",
    "push-object",
    "pull-object",
    "confirm-message-received",
    "push-messages",
    "pull-messages",
    "push-logs",
    "get-nodes",
    "create-task",
    "start-automation",
    "push-task-message",
    "push-task-events",
    "get-run-series-events",
    "pull-task-message",
    "record-task-usage",
    "get-connector",
)
_RESPONSE_NAME_OVERRIDES = {
    "push-messages": "PushAppMessagesResponse",
    "pull-messages": "PullAppMessagesResponse",
}


@pytest.mark.parametrize("response_carrier", ["valid", "invalid", ""])
def test_task_context_round_trips_in_headers(
    monkeypatch: pytest.MonkeyPatch, response_carrier: str
) -> None:
    """Real protobuf HTTP boundaries preserve per-task transport metadata."""
    carrier = "00-" + "1" * 32 + "-" + "2" * 16 + "-03"
    response_carrier = carrier if response_carrier == "valid" else response_carrier
    expected = carrier if response_carrier == carrier else ""
    monkeypatch.setattr(
        import_module("flwr.supercore.routers.runtime.router"),
        "load_task_context",
        lambda state, task_id: response_carrier,
    )
    monkeypatch.setattr(
        "flwr.supercore.runtime.runtime_http_client.current_traceparent",
        lambda: carrier,
    )
    state, handlers = Mock(), Mock()
    task = Task(task_id=5)
    handlers.acquire_task.return_value = AcquireTaskResponse(
        task=task, token="test-token"
    )
    handlers.pull_task_input.return_value = PullTaskInputResponse(task_id=5)
    app = FastAPI()
    app.router.route_class = ProtobufRoute
    app.add_middleware(ProtobufTranslationMiddleware)

    @app.post("/v1/runtime/acquire-task")
    def acquire(
        http_request: Request,
        request: Annotated[AcquireTaskRequest, Depends(get_protobuf_request)],
    ) -> AcquireTaskResponse:
        return acquire_task(http_request, request, state, handlers, None)

    @app.post("/v1/runtime/pull-task-input")
    def pull(
        http_request: Request,
        request: Annotated[PullTaskInputRequest, Depends(get_protobuf_request)],
    ) -> PullTaskInputResponse:
        return pull_task_input(http_request, request, state, handlers, task)

    with (
        TestClient(app) as server,
        RuntimeHttpClient.from_server_address(
            "runtime.example",
            True,
            None,
            interceptors=[RuntimeTokenHttpInterceptor("test-token")],
        ) as client,
    ):

        def send(request: httpx.Request) -> httpx.Response:
            assert request.headers["traceparent"] == carrier
            assert request.headers[TASK_TOKEN_HEADER] == "test-token"
            response = server.post(
                request.url.path, content=request.content, headers=dict(request.headers)
            )
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=response.content,
                request=request,
            )

        with patch.object(
            client._client, "send", side_effect=send  # pylint: disable=protected-access
        ):  # pylint: disable=protected-access
            assert client.AcquireTask(AcquireTaskRequest()).task.task_id == 5
            assert client.take_task_traceparent(6) == ""
            assert client.take_task_traceparent(5) == expected
            assert client.take_task_traceparent(5) == ""
            assert client.PullTaskInput(PullTaskInputRequest()).task_id == 5
            assert client.take_task_traceparent(5) == expected


def test_concurrent_task_response_context_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent calls pair their own carrier with the decoded task ID."""
    active: ContextVar[str] = ContextVar("test_traceparent")
    monkeypatch.setattr(
        "flwr.supercore.runtime.runtime_http_client.current_traceparent", active.get
    )
    barrier = Barrier(2)

    def send(request: httpx.Request) -> httpx.Response:
        carrier = request.headers["traceparent"]
        task_id = int(carrier.split("-")[1], 16)
        barrier.wait(timeout=5)
        return httpx.Response(
            200,
            headers={"traceparent": carrier},
            content=PullTaskInputResponse(task_id=task_id).SerializeToString(),
            request=request,
        )

    with RuntimeHttpClient("http://runtime.example") as client:

        def pull(task_id: int) -> tuple[int, str]:
            active.set(f"00-{task_id:032x}-{task_id:016x}-01")
            result = client.PullTaskInput(PullTaskInputRequest())
            return result.task_id, client.take_task_traceparent(task_id)

        with (
            patch("flwr.supercore.protobuf.client.httpx.Client.send", side_effect=send),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            assert list(pool.map(pull, [5, 6])) == [
                (task_id, f"00-{task_id:032x}-{task_id:016x}-01") for task_id in [5, 6]
            ]


@pytest.mark.parametrize("failure", ["current_traceparent", "validate_traceparent"])
def test_tracing_interceptor_failure_preserves_rpc(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Telemetry failures do not fail successful RPCs or swallow transport errors."""
    monkeypatch.setattr(
        f"flwr.supercore.runtime.runtime_http_client.{failure}",
        Mock(side_effect=RuntimeError("telemetry failure")),
    )
    error = httpx.ConnectError("transport failure")
    with RuntimeHttpClient("http://runtime.example") as client:
        response = httpx.Response(
            200,
            headers={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
            content=PullTaskInputResponse(task_id=5).SerializeToString(),
            request=httpx.Request("POST", "http://runtime.example"),
        )
        with patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
            side_effect=[response, error],
        ) as send:
            assert client.PullTaskInput(PullTaskInputRequest()).task_id == 5
            with pytest.raises(httpx.ConnectError) as caught:
                client.PullTaskInput(PullTaskInputRequest())
            assert caught.value is error
            assert send.call_count == 2


@pytest.mark.parametrize(
    "endpoint",
    _UNARY_UNARY_PATHS,
)
def test_runtime_method(endpoint: str) -> None:
    """Call one shared Runtime HTTP endpoint."""
    method_name = endpoint.title().replace("-", "")
    request = Mock()
    response = Mock()
    client = RuntimeHttpClient("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = getattr(client, method_name)(request)

    assert result is response
    call.assert_called_once()
    assert call.call_args.kwargs["path"] == f"/v1/runtime/{endpoint}"
    assert call.call_args.kwargs["rpc_method"] == f"/flwr.proto.Runtime/{method_name}"
    assert call.call_args.kwargs["request"] is request
    expected_response_name = _RESPONSE_NAME_OVERRIDES.get(
        endpoint, f"{method_name}Response"
    )
    assert call.call_args.kwargs["response_type"].__name__ == expected_response_name
