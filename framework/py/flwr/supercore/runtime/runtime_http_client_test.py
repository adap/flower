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

from importlib import import_module
from typing import Annotated, cast
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    AcquireTaskResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.dependencies.runtime import get_runtime_state, get_task
from flwr.supercore.interceptors import RuntimeTokenHttpInterceptor
from flwr.supercore.protobuf.client import ProtobufClient
from flwr.supercore.protobuf.routing import ProtobufRoute
from flwr.supercore.protobuf.translation import (
    ProtobufTranslationMiddleware,
    get_protobuf_request,
)
from flwr.supercore.routers.runtime.router import acquire_task, pull_task_input, router
from flwr.supercore.routers.runtime.tracing import RuntimeTracingMiddleware
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.runtime.tracing import RuntimeTracingInterceptor

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


def test_task_context_round_trips_in_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real protobuf HTTP boundaries preserve per-task transport metadata."""
    carrier = "00-" + "1" * 32 + "-" + "2" * 16 + "-03"
    monkeypatch.setattr(
        import_module("flwr.supercore.routers.runtime.tracing"),
        "load_task_context",
        lambda state, task_id: carrier,
    )
    monkeypatch.setattr(
        "flwr.supercore.runtime.tracing.current_traceparent",
        lambda: carrier,
    )
    state, handlers = Mock(), Mock()
    task = Task(task_id=5)
    handlers.acquire_task.return_value = AcquireTaskResponse(
        task=task, token="test-token"
    )
    handlers.pull_task_input.return_value = PullTaskInputResponse(task_id=5)
    app = FastAPI()
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    app.state.runtime_state_factory = Mock(state=lambda: state)
    app.router.route_class = ProtobufRoute
    app.add_middleware(RuntimeTracingMiddleware)
    app.add_middleware(ProtobufTranslationMiddleware)

    @app.post("/v1/runtime/acquire-task")
    def acquire(
        request: Annotated[AcquireTaskRequest, Depends(get_protobuf_request)],
    ) -> AcquireTaskResponse:
        return acquire_task(request, state, handlers, None)

    @app.post("/v1/runtime/pull-task-input")
    def pull(
        request: Annotated[PullTaskInputRequest, Depends(get_protobuf_request)],
    ) -> PullTaskInputResponse:
        return pull_task_input(request, state, handlers, task)

    tracing = RuntimeTracingInterceptor()
    with (
        TestClient(app) as server,
        RuntimeHttpClient.from_server_address(
            "runtime.example",
            True,
            None,
            interceptors=[tracing, RuntimeTokenHttpInterceptor("test-token")],
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
            assert tracing.take_task_traceparent(6) == ""
            assert tracing.take_task_traceparent(5) == carrier
            assert tracing.take_task_traceparent(5) == ""
            assert client.PullTaskInput(PullTaskInputRequest()).task_id == 5
            assert tracing.take_task_traceparent(5) == carrier
            app.state.runtime_state_factory.state = Mock(
                side_effect=RuntimeError("metadata")
            )
            assert client.AcquireTask(AcquireTaskRequest()).task.task_id == 5
            assert tracing.take_task_traceparent(5) == ""


@pytest.mark.parametrize("failure", ["current_traceparent", "validate_traceparent"])
def test_tracing_interceptor_failure_preserves_rpc(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Telemetry failures do not fail successful RPCs or swallow transport errors."""
    monkeypatch.setattr(
        f"flwr.supercore.runtime.tracing.{failure}",
        Mock(side_effect=RuntimeError("telemetry failure")),
    )
    error = httpx.ConnectError("transport failure")
    with RuntimeHttpClient(
        "http://runtime.example", interceptors=[RuntimeTracingInterceptor()]
    ) as client:
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


@pytest.mark.parametrize("failed", [False, True])
def test_completion_tracing_scopes_handler_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    """The dependency wraps business work even if the metadata lookup fails."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    manager = MagicMock()
    backend = Mock(trace_span=Mock(return_value=manager))
    backend.load_task_context.side_effect = RuntimeError("metadata")
    monkeypatch.setattr("flwr.supercore.tracing._load_backend", lambda: backend)
    handlers = Mock()
    error = ValueError("business failure")

    def finish(*_args: object) -> PushTaskOutputResponse:
        manager.__enter__.assert_called_once()
        manager.__exit__.assert_not_called()
        if failed:
            raise error
        return PushTaskOutputResponse()

    handlers.push_task_output.side_effect = finish
    state = Mock()
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(ProtobufTranslationMiddleware)
    app.state.runtime_handlers = handlers
    app.dependency_overrides[get_runtime_state] = lambda: state
    app.dependency_overrides[get_task] = lambda: Task(task_id=5, run_id=7)
    with TestClient(app) as server:

        def push() -> httpx.Response:
            return cast(
                httpx.Response,
                server.post(
                    "/v1/runtime/push-task-output",
                    content=PushTaskOutputRequest().SerializeToString(),
                    headers={"content-type": "application/protobuf"},
                ),
            )

        if failed:
            with pytest.raises(ValueError) as caught:
                push()
            assert caught.value is error
        else:
            assert push().status_code == 200
        manager.__exit__.assert_called_once_with(None, None, None)


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
