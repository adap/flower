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
"""Flower ClientApp process."""


import argparse
import gc
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from logging import DEBUG, ERROR, INFO
from typing import Any, Optional

import grpc
from tqdm import tqdm

from flwr.cli.install import install_from_fab
from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.common import Context, Message
from flwr.common.args import add_args_flwr_app_common
from flwr.common.config import get_flwr_dir
from flwr.common.constant import (
    CHUNK_SIZE,
    CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    ErrorCode,
)
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.message import (
    Error,
    allocate_byte_arrays,
    chunk_viewer,
    decouple_arrays_from_message,
    materialize_arrays,
    total_num_chunks,
)
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    message_from_proto,
    message_to_proto,
    run_from_proto,
)
from flwr.common.typing import Fab, Run
from flwr.proto.chunk_pb2 import (  # pylint: disable=E0611
    Chunk,
    PullChunkRequest,
    PullChunkResponse,
    PushChunkRequest,
    PushChunkResponse,
)

# pylint: disable=E0611
from flwr.proto.clientappio_pb2 import (
    GetTokenRequest,
    GetTokenResponse,
    PullClientAppInputsRequest,
    PullClientAppInputsResponse,
    PushClientAppOutputsRequest,
    PushClientAppOutputsResponse,
    QueryMessageIdRequest,
)
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub

from .utils import get_load_client_app_fn


def flwr_clientapp() -> None:
    """Run process-isolated Flower ClientApp."""
    args = _parse_args_run_flwr_clientapp().parse_args()
    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "flwr-clientapp does not support TLS yet.",
        )

    log(INFO, "Start `flwr-clientapp` process")
    log(
        DEBUG,
        "`flwr-clientapp` will attempt to connect to SuperNode's "
        "ClientAppIo API at %s with token %s",
        args.clientappio_api_address,
        args.token,
    )
    run_clientapp(
        clientappio_api_address=args.clientappio_api_address,
        run_once=(args.token is not None),
        token=args.token,
        flwr_dir=args.flwr_dir,
        certificates=None,
    )


def run_clientapp(  # pylint: disable=R0914
    clientappio_api_address: str,
    run_once: bool,
    token: Optional[int] = None,
    flwr_dir: Optional[str] = None,
    certificates: Optional[bytes] = None,
) -> None:
    """Run Flower ClientApp process."""
    channel = create_channel(
        server_address=clientappio_api_address,
        insecure=(certificates is None),
        root_certificates=certificates,
    )
    channel.subscribe(on_channel_state_change)

    # Resolve directory where FABs are installed
    flwr_dir_ = get_flwr_dir(flwr_dir)
    try:
        stub = ClientAppIoStub(channel)
        _wrap_stub(stub, _make_simple_grpc_retry_invoker())

        while True:
            # If token is not set, loop until token is received from SuperNode
            while token is None:
                token = get_token(stub)
                time.sleep(1)

            # Pull Message, Context, Run and (optional) FAB from SuperNode
            message, context, run, fab = pull_clientappinputs(stub=stub, token=token)

            # Allocate bytearrays
            bytearrays_dict = allocate_byte_arrays(msg_content=message.content)
            # Identify number of total chunks based on Array sizes
            total_chunks = total_num_chunks(msg_content=message.content)
            # Request one chunk and store in bytearray
            if total_chunks:
                # Request one chunk at a time
                log(INFO, f"Requesting {total_chunks} chunks!")

            inflight_futures = set()
            num_pulled_chunks = 0
            with (
                ThreadPoolExecutor(max_workers=4) as executor,
                tqdm(total=total_chunks, desc="PullChunk") as pbar,
            ):

                # Submit one request
                future = executor.submit(
                    stub.PullChunk,
                    request=PullChunkRequest(
                        message_id=message.metadata.message_id, node=None
                    ),
                )
                inflight_futures.add(future)

                while num_pulled_chunks < total_chunks:
                    done, inflight_futures = wait(
                        inflight_futures, return_when=FIRST_COMPLETED
                    )

                    for future in done:
                        response: PullChunkResponse = future.result()
                        if (
                            response.chunk.record_id
                        ):  # will be unset if a chunk wasn't available to pull
                            chunk: Chunk = response.chunk
                            # Place memory in the Array it belongs to
                            offset = CHUNK_SIZE * chunk.chunk_index
                            bytearrays_dict[chunk.record_id][chunk.array_id][
                                offset : offset + len(chunk.data)
                            ] = chunk.data
                            num_pulled_chunks += 1
                            pbar.update(1)  # we got a chunk, update progress bar

                        # Submit a new request if still needed
                        if num_pulled_chunks < total_chunks:
                            future_ = executor.submit(
                                stub.PullChunk,
                                request=PullChunkRequest(
                                    message_id=message.metadata.message_id, node=None
                                ),
                            )
                            inflight_futures.add(future_)
                        else:
                            break

                # Put data in Message (i.e. materialize Message)
                materialize_arrays(
                    msg_content=message.content, bytearray_dict=bytearrays_dict
                )

            # Install FAB, if provided
            if fab:
                log(DEBUG, "[flwr-clientapp] Start FAB installation.")
                install_from_fab(fab.content, flwr_dir=flwr_dir_, skip_prompt=True)

            load_client_app_fn = get_load_client_app_fn(
                default_app_ref="",
                app_path=None,
                multi_app=True,
                flwr_dir=str(flwr_dir_),
            )

            try:
                # Load ClientApp
                log(DEBUG, "[flwr-clientapp] Start `ClientApp` Loading.")
                client_app: ClientApp = load_client_app_fn(
                    run.fab_id, run.fab_version, fab.hash_str if fab else ""
                )

                # Execute ClientApp
                reply_message = client_app(message=message, context=context)

            except Exception as ex:  # pylint: disable=broad-exception-caught
                # Don't update/change NodeState

                e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
                # Ex fmt: "<class 'ZeroDivisionError'>:<'division by zero'>"
                reason = str(type(ex)) + ":<'" + str(ex) + "'>"
                exc_entity = "ClientApp"
                if isinstance(ex, LoadClientAppError):
                    reason = (
                        "An exception was raised when attempting to load `ClientApp`"
                    )
                    e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION

                log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)

                # Create error message
                reply_message = Message(
                    Error(code=e_code, reason=reason), reply_to=message
                )

            # Decouple Array data from rest of the Message
            reply_msg, array_records = decouple_arrays_from_message(reply_message)

            # Push Message and Context to SuperNode
            _ = push_clientappoutputs(
                stub=stub, token=token, message=reply_msg, context=context
            )

            # Wait until Message is pushed by SuperNode. Only then the id the message
            # got assigned by the superlink will be accesible via the ClientAppIo API.
            # We need it to construct our PushChunkRequests
            while True:
                msg_id = stub.QueryMessageId(
                    request=QueryMessageIdRequest(msg_hash=reply_msg.hash())
                ).msg_id
                if msg_id == "":
                    time.sleep(0.5)
                else:
                    break

            # Send chunks
            chunk_views = chunk_viewer(array_records)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        _push_chunk, stub=stub, chunk_view=chunk_view, message_id=msg_id
                    )
                    for chunk_view in chunk_views
                ]

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="PushChunk"
                ):
                    _ = future.result()

            del client_app, message, context, run, fab, reply_message
            gc.collect()

            # Reset token to `None` to prevent flwr-clientapp from trying to pull the
            # same inputs again
            token = None

            # Stop the loop if `flwr-clientapp` is expected to process only a single
            # message
            if run_once:
                break

    except KeyboardInterrupt:
        log(INFO, "Closing connection")
    except grpc.RpcError as e:
        log(ERROR, "GRPC error occurred: %s", str(e))
    finally:
        channel.close()


def _push_chunk(
    stub, message_id: str, chunk_view: list[dict[str, Any]]
) -> PushChunkResponse:

    # materialize Chunk
    chunk = Chunk(
        array_id=chunk_view["array_id"],
        record_id=chunk_view["record_id"],
        chunk_index=chunk_view["chunk_index"],
        data=chunk_view["data"].tobytes(),  # <----- materialize chunk (copies data)
    )
    # Push Chunk
    _: PushChunkResponse = stub.PushChunk(
        PushChunkRequest(chunks=[chunk], message_id=message_id)
    )


def get_token(stub: grpc.Channel) -> Optional[int]:
    """Get a token from SuperNode."""
    log(DEBUG, "[flwr-clientapp] Request token")
    try:
        res: GetTokenResponse = stub.GetToken(GetTokenRequest())
        log(DEBUG, "[GetToken] Received token: %s", res.token)
        return res.token
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.FAILED_PRECONDITION:  # pylint: disable=no-member
            log(DEBUG, "[GetToken] No token available yet")
        else:
            log(ERROR, "[GetToken] gRPC error occurred: %s", str(e))
        return None


def pull_clientappinputs(
    stub: grpc.Channel, token: int
) -> tuple[Message, Context, Run, Optional[Fab]]:
    """Pull ClientAppInputs from SuperNode."""
    log(INFO, "[flwr-clientapp] Pull `ClientAppInputs` for token %s", token)
    try:
        res: PullClientAppInputsResponse = stub.PullClientAppInputs(
            PullClientAppInputsRequest(token=token)
        )
        message = message_from_proto(res.message)
        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab) if res.fab else None
        return message, context, run, fab
    except grpc.RpcError as e:
        log(ERROR, "[PullClientAppInputs] gRPC error occurred: %s", str(e))
        raise e


def push_clientappoutputs(
    stub: grpc.Channel, token: int, message: Message, context: Context
) -> PushClientAppOutputsResponse:
    """Push ClientAppOutputs to SuperNode."""
    log(INFO, "[flwr-clientapp] Push `ClientAppOutputs` for token %s", token)
    proto_message = message_to_proto(message)
    proto_context = context_to_proto(context)

    try:
        res: PushClientAppOutputsResponse = stub.PushClientAppOutputs(
            PushClientAppOutputsRequest(
                token=token, message=proto_message, context=proto_context
            )
        )
        return res
    except grpc.RpcError as e:
        log(ERROR, "[PushClientAppOutputs] gRPC error occurred: %s", str(e))
        raise e


def _parse_args_run_flwr_clientapp() -> argparse.ArgumentParser:
    """Parse flwr-clientapp command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower ClientApp",
    )
    parser.add_argument(
        "--clientappio-api-address",
        default=CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperNode's ClientAppIo API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS}.",
    )
    parser.add_argument(
        "--token",
        type=int,
        required=False,
        help="Unique token generated by SuperNode for each ClientApp execution",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
