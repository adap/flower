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


import gc
import os
import threading
import time
from logging import DEBUG, ERROR, INFO
from typing import Optional

import grpc

from flwr.app.error import Error
from flwr.cli.install import install_from_fab
from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.client.clientapp.utils import get_load_client_app_fn
from flwr.common import Context, Message
from flwr.common.config import get_flwr_dir
from flwr.common.constant import ErrorCode
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.inflatable import (
    get_all_nested_objects,
    get_object_tree,
    no_object_id_recompute,
)
from flwr.common.inflatable_protobuf_utils import (
    make_confirm_message_received_fn_protobuf,
    make_pull_object_fn_protobuf,
    make_push_object_fn_protobuf,
)
from flwr.common.inflatable_utils import pull_and_inflate_object_from_tree, push_objects
from flwr.common.logger import log
from flwr.common.message import remove_content_from_message
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    message_to_proto,
    run_from_proto,
)
from flwr.common.typing import Fab, Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppInputsRequest,
    PullAppInputsResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
)

# pylint: disable=E0611
from flwr.proto.clientappio_pb2 import (
    GetRunIdsWithPendingMessagesRequest,
    GetRunIdsWithPendingMessagesResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.supercore.utils import mask_string


def run_clientapp(  # pylint: disable=R0913, R0914, R0917
    clientappio_api_address: str,
    run_once: bool,
    token: Optional[str] = None,
    flwr_dir: Optional[str] = None,
    certificates: Optional[bytes] = None,
    parent_pid: Optional[int] = None,
) -> None:
    """Run Flower ClientApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

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
            if token is None:
                token = get_token(stub)

            # Pull Message, Context, Run and (optional) FAB from SuperNode
            message, context, run, fab = pull_clientappinputs(stub=stub, token=token)

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

            # Push Message and Context to SuperNode
            _ = push_clientappoutputs(
                stub=stub, token=token, message=reply_message, context=context
            )

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


def start_parent_process_monitor(
    parent_pid: int,
) -> None:
    """Monitor the parent process and exit if it terminates."""

    def monitor() -> None:
        while True:
            time.sleep(0.2)
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), 9)

    threading.Thread(target=monitor, daemon=True).start()


def get_token(stub: ClientAppIoStub) -> str:
    """Get a token from SuperNode."""
    log(DEBUG, "[flwr-clientapp] Request token")
    while True:
        res: GetRunIdsWithPendingMessagesResponse = stub.GetRunIdsWithPendingMessages(
            GetRunIdsWithPendingMessagesRequest()
        )

        for run_id in res.run_ids:
            tk_res: RequestTokenResponse = stub.RequestToken(
                RequestTokenRequest(run_id=run_id)
            )
            if tk_res.token:
                return tk_res.token

        time.sleep(1)  # Wait before retrying to get run IDs


def pull_clientappinputs(
    stub: ClientAppIoStub, token: str
) -> tuple[Message, Context, Run, Optional[Fab]]:
    """Pull ClientAppInputs from SuperNode."""
    masked_token = mask_string(token)
    log(INFO, "[flwr-clientapp] Pull `ClientAppInputs` for token %s", masked_token)
    try:
        # Pull Context, Run and (optional) FAB
        res: PullAppInputsResponse = stub.PullClientAppInputs(
            PullAppInputsRequest(token=token)
        )
        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab) if res.fab else None

        # Pull and inflate the message
        pull_msg_res: PullAppMessagesResponse = stub.PullMessage(
            PullAppMessagesRequest(token=token)
        )
        run_id = context.run_id
        node = Node(node_id=context.node_id)
        object_tree = pull_msg_res.message_object_trees[0]
        message = pull_and_inflate_object_from_tree(
            object_tree,
            make_pull_object_fn_protobuf(stub.PullObject, node, run_id),
            make_confirm_message_received_fn_protobuf(
                stub.ConfirmMessageReceived, node, run_id
            ),
            return_type=Message,
        )

        # Set the message ID
        # The deflated message doesn't contain the message_id (its own object_id)
        message.metadata.__dict__["_message_id"] = object_tree.object_id
        return message, context, run, fab
    except grpc.RpcError as e:
        log(ERROR, "[PullClientAppInputs] gRPC error occurred: %s", str(e))
        raise e


def push_clientappoutputs(
    stub: ClientAppIoStub, token: str, message: Message, context: Context
) -> PushAppOutputsResponse:
    """Push ClientAppOutputs to SuperNode."""
    masked_token = mask_string(token)
    log(INFO, "[flwr-clientapp] Push `ClientAppOutputs` for token %s", masked_token)
    # Set message ID
    message.metadata.__dict__["_message_id"] = message.object_id
    proto_message = message_to_proto(remove_content_from_message(message))
    proto_context = context_to_proto(context)

    try:

        with no_object_id_recompute():
            # Get object tree and all objects to push
            object_tree = get_object_tree(message)

            # Push Message
            # This is temporary. The message should not contain its content
            push_msg_res = stub.PushMessage(
                PushAppMessagesRequest(
                    token=token,
                    messages_list=[proto_message],
                    message_object_trees=[object_tree],
                )
            )
            del proto_message

            # Retrieve the object IDs to push
            object_ids_to_push = set(
                push_msg_res.objects_to_push[object_tree.object_id].object_ids
            )

            # Push all objects
            all_objects = get_all_nested_objects(message)
            del message
            push_objects(
                all_objects,
                make_push_object_fn_protobuf(
                    stub.PushObject,
                    Node(node_id=context.node_id),
                    run_id=context.run_id,
                ),
                object_ids_to_push=object_ids_to_push,
            )

        # Push Context
        res: PushAppOutputsResponse = stub.PushClientAppOutputs(
            PushAppOutputsRequest(token=token, context=proto_context)
        )
        return res
    except grpc.RpcError as e:
        log(ERROR, "[PushClientAppOutputs] gRPC error occurred: %s", str(e))
        raise e
