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
"""SuperExec API servicer."""


import time
from collections.abc import Generator
from logging import ERROR, INFO
from typing import Any, Optional, cast

import grpc

from flwr.common import now
from flwr.common.auth_plugin import ExecAuthPlugin
from flwr.common.constant import (
    FAB_MAX_SIZE,
    LOG_STREAM_INTERVAL,
    RUN_ID_NOT_FOUND_MESSAGE,
    Status,
    SubStatus,
)
from flwr.common.logger import log
from flwr.common.serde import (
    config_record_from_proto,
    run_to_proto,
    user_config_from_proto,
)
from flwr.common.typing import Run, RunStatus
from flwr.proto import exec_pb2_grpc  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    ListRunsRequest,
    ListRunsResponse,
    StartRunRequest,
    StartRunResponse,
    StopRunRequest,
    StopRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStore, ObjectStoreFactory

from .exec_user_auth_interceptor import shared_account_info
from .executor import Executor


class ExecServicer(exec_pb2_grpc.ExecServicer):
    """SuperExec API servicer."""

    def __init__(  # pylint: disable=R0913, R0917
        self,
        linkstate_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
        executor: Executor,
        auth_plugin: Optional[ExecAuthPlugin] = None,
    ) -> None:
        self.linkstate_factory = linkstate_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory
        self.executor = executor
        self.executor.initialize(linkstate_factory, ffs_factory)
        self.auth_plugin = auth_plugin

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ExecServicer.StartRun")

        if len(request.fab.content) > FAB_MAX_SIZE:
            log(
                ERROR,
                "FAB size exceeds maximum allowed size of %d bytes.",
                FAB_MAX_SIZE,
            )
            return StartRunResponse()

        flwr_aid = shared_account_info.get().flwr_aid if self.auth_plugin else None
        run_id = self.executor.start_run(
            request.fab.content,
            user_config_from_proto(request.override_config),
            config_record_from_proto(request.federation_options),
            flwr_aid,
        )

        if run_id is None:
            log(ERROR, "Executor failed to start run")
            return StartRunResponse()

        return StartRunResponse(run_id=run_id)

    def StreamLogs(  # pylint: disable=C0103
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, "ExecServicer.StreamLogs")
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        run = state.get_run(run_id)

        # Exit if `run_id` not found
        if not run:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)

        # If user auth is enabled, check if `flwr_aid` matches the run's `flwr_aid`
        if self.auth_plugin:
            flwr_aid = shared_account_info.get().flwr_aid
            _check_flwr_aid_in_run(
                flwr_aid=flwr_aid, run=cast(Run, run), context=context
            )

        after_timestamp = request.after_timestamp + 1e-6
        while context.is_active():
            log_msg, latest_timestamp = state.get_serverapp_log(run_id, after_timestamp)
            if log_msg:
                yield StreamLogsResponse(
                    log_output=log_msg,
                    latest_timestamp=latest_timestamp,
                )
                # Add a small epsilon to the latest timestamp to avoid getting
                # the same log
                after_timestamp = max(latest_timestamp + 1e-6, after_timestamp)

            # Wait for and continue to yield more log responses only if the
            # run isn't completed yet. If the run is finished, the entire log
            # is returned at this point and the server ends the stream.
            run_status = state.get_run_status({run_id})[run_id]
            if run_status.status == Status.FINISHED:
                log(INFO, "All logs for run ID `%s` returned", run_id)

                # Delete objects of the run from the object store
                self.objectstore_factory.store().delete_objects_in_run(run_id)
                break

            time.sleep(LOG_STREAM_INTERVAL)  # Sleep briefly to avoid busy waiting

    def ListRuns(
        self, request: ListRunsRequest, context: grpc.ServicerContext
    ) -> ListRunsResponse:
        """Handle `flwr ls` command."""
        log(INFO, "ExecServicer.List")
        state = self.linkstate_factory.state()

        # Build a set of run IDs for `flwr ls --runs`
        if not request.HasField("run_id"):
            if self.auth_plugin:
                # If no `run_id` is specified and user auth is enabled,
                # return run IDs for the authenticated user
                flwr_aid = shared_account_info.get().flwr_aid
                if flwr_aid is None:
                    context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        "️⛔️ User authentication is enabled, but `flwr_aid` is None",
                    )
                run_ids = state.get_run_ids(flwr_aid=flwr_aid)
            else:
                # If no `run_id` is specified and no user auth is enabled,
                # return all run IDs
                run_ids = state.get_run_ids(None)
        # Build a set of run IDs for `flwr ls --run-id <run_id>`
        else:
            # Retrieve run ID and run
            run_id = request.run_id
            run = state.get_run(run_id)

            # Exit if `run_id` not found
            if not run:
                context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)

            # If user auth is enabled, check if `flwr_aid` matches the run's `flwr_aid`
            if self.auth_plugin:
                flwr_aid = shared_account_info.get().flwr_aid
                _check_flwr_aid_in_run(
                    flwr_aid=flwr_aid, run=cast(Run, run), context=context
                )

            run_ids = {run_id}

        # Init the object store
        store = self.objectstore_factory.store()
        return _create_list_runs_response(run_ids, state, store)

    def StopRun(
        self, request: StopRunRequest, context: grpc.ServicerContext
    ) -> StopRunResponse:
        """Stop a given run ID."""
        log(INFO, "ExecServicer.StopRun")
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        run = state.get_run(run_id)

        # Exit if `run_id` not found
        if not run:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)

        # If user auth is enabled, check if `flwr_aid` matches the run's `flwr_aid`
        if self.auth_plugin:
            flwr_aid = shared_account_info.get().flwr_aid
            _check_flwr_aid_in_run(
                flwr_aid=flwr_aid, run=cast(Run, run), context=context
            )

        run_status = state.get_run_status({run_id})[run_id]
        if run_status.status == Status.FINISHED:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Run ID {run_id} is already finished",
            )

        update_success = state.update_run_status(
            run_id=run_id,
            new_status=RunStatus(Status.FINISHED, SubStatus.STOPPED, ""),
        )

        if update_success:
            message_ids: set[str] = state.get_message_ids_from_run_id(run_id)

            # Delete Messages and their replies for the `run_id`
            state.delete_messages(message_ids)

            # Delete objects of the run from the object store
            self.objectstore_factory.store().delete_objects_in_run(run_id)

        return StopRunResponse(success=update_success)

    def GetLoginDetails(
        self, request: GetLoginDetailsRequest, context: grpc.ServicerContext
    ) -> GetLoginDetailsResponse:
        """Start login."""
        log(INFO, "ExecServicer.GetLoginDetails")
        if self.auth_plugin is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "ExecServicer initialized without user authentication",
            )
            raise grpc.RpcError()  # This line is unreachable

        # Get login details
        details = self.auth_plugin.get_login_details()

        # Return empty response if details is None
        if details is None:
            return GetLoginDetailsResponse()

        return GetLoginDetailsResponse(
            auth_type=details.auth_type,
            device_code=details.device_code,
            verification_uri_complete=details.verification_uri_complete,
            expires_in=details.expires_in,
            interval=details.interval,
        )

    def GetAuthTokens(
        self, request: GetAuthTokensRequest, context: grpc.ServicerContext
    ) -> GetAuthTokensResponse:
        """Get auth token."""
        log(INFO, "ExecServicer.GetAuthTokens")
        if self.auth_plugin is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "ExecServicer initialized without user authentication",
            )
            raise grpc.RpcError()  # This line is unreachable

        # Get auth tokens
        credentials = self.auth_plugin.get_auth_tokens(request.device_code)

        # Return empty response if credentials is None
        if credentials is None:
            return GetAuthTokensResponse()

        return GetAuthTokensResponse(
            access_token=credentials.access_token,
            refresh_token=credentials.refresh_token,
        )


def _create_list_runs_response(
    run_ids: set[int], state: LinkState, store: ObjectStore
) -> ListRunsResponse:
    """Create response for `flwr ls --runs` and `flwr ls --run-id <run_id>`."""
    run_dict = {run_id: run for run_id in run_ids if (run := state.get_run(run_id))}

    # Delete objects of finished runs from the object store
    for run_id, run in run_dict.items():
        if run.status.status == Status.FINISHED:
            store.delete_objects_in_run(run_id)

    return ListRunsResponse(
        run_dict={run_id: run_to_proto(run) for run_id, run in run_dict.items()},
        now=now().isoformat(),
    )


def _check_flwr_aid_in_run(
    flwr_aid: Optional[str], run: Run, context: grpc.ServicerContext
) -> None:
    """Guard clause to check if `flwr_aid` matches the run's `flwr_aid`."""
    # `flwr_aid` must not be None. Abort if it is None.
    if flwr_aid is None:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "️⛔️ User authentication is enabled, but `flwr_aid` is None",
        )

    # `run.flwr_aid` must not be an empty string. Abort if it is empty.
    run_flwr_aid = run.flwr_aid
    if not run_flwr_aid:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "⛔️ User authentication is enabled, but the run is not associated "
            "with a `flwr_aid`.",
        )

    # Exit if `flwr_aid` does not match the run's `flwr_aid`
    if run_flwr_aid != flwr_aid:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "⛔️ Run ID does not belong to the user",
        )
