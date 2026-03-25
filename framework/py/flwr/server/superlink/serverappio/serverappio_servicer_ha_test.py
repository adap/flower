# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""HA-focused ServerAppIoServicer tests."""


import os
import tempfile
import threading

import grpc

from flwr.common import ConfigRecord, Context, RecordDict
from flwr.common.constant import SUPERLINK_NODE_ID, Status
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppInputsRequest,
    PullAppInputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.server.superlink.linkstate.linkstate import LinkState
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc
from flwr.supercore.constant import NOOP_FEDERATION, RunType
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


def _start_serverappio_with_port_retry(
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    objectstore_factory: ObjectStoreFactory,
    start_port: int,
) -> grpc.Server:
    for offset in range(40):
        address = f"127.0.0.1:{start_port + offset}"
        try:
            return run_serverappio_api_grpc(
                address,
                state_factory,
                ffs_factory,
                objectstore_factory,
                None,
            )
        except RuntimeError as err:
            if "Failed to bind to address" in str(err):
                continue
            raise

    raise AssertionError(
        f"Could not bind ServerAppIo gRPC server starting at port {start_port}."
    )


def _create_shared_runtime(
    tmpdir: str,
) -> tuple[int, LinkState, grpc.Server, grpc.Server]:
    database_path = os.path.join(tmpdir, "shared.db")
    storage_dir = os.path.join(tmpdir, "ffs")
    os.makedirs(storage_dir, exist_ok=True)

    objectstore_factory_0 = ObjectStoreFactory()
    objectstore_factory_1 = ObjectStoreFactory()
    state_factory_0 = LinkStateFactory(
        database_path, NoOpFederationManager(), objectstore_factory_0
    )
    state_factory_1 = LinkStateFactory(
        database_path, NoOpFederationManager(), objectstore_factory_1
    )
    state_0 = state_factory_0.state()
    ffs_factory_0 = FfsFactory(storage_dir)
    ffs_factory_1 = FfsFactory(storage_dir)
    fab_hash = ffs_factory_0.ffs().put(b"mock fab content", {})

    run_id = state_0.create_run(
        "",
        "",
        fab_hash,
        {},
        NOOP_FEDERATION,
        ConfigRecord(),
        "",
        RunType.SERVER_APP,
    )
    state_0.set_serverapp_context(
        run_id, Context(run_id, SUPERLINK_NODE_ID, {}, RecordDict(), {})
    )
    server_0 = _start_serverappio_with_port_retry(
        state_factory_0,
        ffs_factory_0,
        objectstore_factory_0,
        start_port=19091,
    )
    server_1 = _start_serverappio_with_port_retry(
        state_factory_1,
        ffs_factory_1,
        objectstore_factory_1,
        start_port=19141,
    )
    return run_id, state_0, server_0, server_1


def _request_token(channel: grpc.Channel, run_id: int) -> str:
    request_token = channel.unary_unary(
        "/flwr.proto.ServerAppIo/RequestToken",
        request_serializer=RequestTokenRequest.SerializeToString,
        response_deserializer=RequestTokenResponse.FromString,
    )
    token_response, token_call = request_token.with_call(
        RequestTokenRequest(run_id=run_id)
    )
    assert grpc.StatusCode.OK == token_call.code()
    token = str(token_response.token)
    assert token
    return token


def _claim_in_parallel(
    channel_0: grpc.Channel, channel_1: grpc.Channel, token: str
) -> list[grpc.StatusCode | None]:
    pull_app_inputs_0 = channel_0.unary_unary(
        "/flwr.proto.ServerAppIo/PullAppInputs",
        request_serializer=PullAppInputsRequest.SerializeToString,
        response_deserializer=PullAppInputsResponse.FromString,
    )
    pull_app_inputs_1 = channel_1.unary_unary(
        "/flwr.proto.ServerAppIo/PullAppInputs",
        request_serializer=PullAppInputsRequest.SerializeToString,
        response_deserializer=PullAppInputsResponse.FromString,
    )
    timeout = 5.0
    barrier = threading.Barrier(3)
    results: list[grpc.StatusCode | None] = [None, None]
    exceptions: list[Exception] = []

    def claim_inputs(idx: int, pull_fn: grpc.UnaryUnaryMultiCallable) -> None:
        try:
            barrier.wait(timeout=timeout)
            response, call = pull_fn.with_call(PullAppInputsRequest(token=token))
            del response
            results[idx] = call.code()
        except grpc.RpcError as err:
            results[idx] = err.code()
        except Exception as ex:  # pylint: disable=broad-exception-caught
            exceptions.append(ex)

    threads = [
        threading.Thread(target=claim_inputs, args=(0, pull_app_inputs_0)),
        threading.Thread(target=claim_inputs, args=(1, pull_app_inputs_1)),
    ]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=timeout)
    except threading.BrokenBarrierError as ex:
        exceptions.append(ex)
    for thread in threads:
        thread.join(timeout=timeout)

    alive_threads = [thread for thread in threads if thread.is_alive()]
    if alive_threads:
        raise AssertionError(
            f"Concurrent PullAppInputs test timed out; {len(alive_threads)} "
            f"thread(s) still alive after {timeout} seconds."
        )
    if exceptions:
        raise exceptions[0]
    return results


def test_pull_app_inputs_claim_is_unique_across_replicas() -> None:
    """Ensure only one replica can claim STARTING -> RUNNING via PullAppInputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id, state_0, server_0, server_1 = _create_shared_runtime(tmpdir)
        channel_0 = grpc.insecure_channel(server_0.bound_address)
        channel_1 = grpc.insecure_channel(server_1.bound_address)
        try:
            token = _request_token(channel_0, run_id)
            results = _claim_in_parallel(channel_0, channel_1, token)

            assert results.count(grpc.StatusCode.OK) == 1
            assert results.count(grpc.StatusCode.FAILED_PRECONDITION) == 1
            run_status = state_0.get_run_status({run_id})[run_id]
            assert run_status.status == Status.RUNNING
        finally:
            channel_0.close()
            channel_1.close()
            server_0.stop(None)
            server_1.stop(None)
