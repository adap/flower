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
"""Shared authentication helpers for AppIO servicers."""


import grpc

from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.superexec_auth import (
    SuperExecAuthConfig,
    superexec_auth_metadata_present,
    verify_superexec_signed_metadata,
)


class AppIoAuthMixin:
    """Shared authentication methods for AppIO servicers."""

    state_factory: LinkStateFactory
    superexec_auth_config: SuperExecAuthConfig
    _EXEC_PLUGIN_TYPE: str

    def _verify_token(self, token: str, context: grpc.ServicerContext) -> int:
        """Verify the token and return the associated run ID."""
        state = self.state_factory.state()
        run_id = state.get_run_id_by_token(token)
        if run_id is None or not state.verify_token(run_id, token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")
        return run_id

    def _verify_token_for_run(
        self, token: str, run_id: int, context: grpc.ServicerContext
    ) -> None:
        """Verify token and ensure it belongs to the provided run ID."""
        token_run_id = self._verify_token(token, context)
        if token_run_id != run_id:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token for run ID.",
            )
            raise RuntimeError("This line should never be reached.")

    def _verify_superexec_auth_if_enabled(
        self, context: grpc.ServicerContext, method: str
    ) -> None:
        """Verify SuperExec signed metadata when SuperExec auth is enabled."""
        if not self.superexec_auth_config.enabled:
            return
        verify_superexec_signed_metadata(
            context=context,
            method=method,
            plugin_type=self._EXEC_PLUGIN_TYPE,
            cfg=self.superexec_auth_config,
        )

    def _verify_get_run_auth_if_enabled(
        self,
        token: str,
        run_id: int,
        context: grpc.ServicerContext,
        method: str,
    ) -> None:
        """Authorize GetRun with one mechanism when SuperExec auth is enabled."""
        if not self.superexec_auth_config.enabled:
            # Legacy behavior by design: when SuperExec auth is disabled, GetRun
            # remains unauthenticated and tokenless requests are allowed.
            return

        token_present = bool(token)
        signed_metadata_present = superexec_auth_metadata_present(context)
        if token_present == signed_metadata_present:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Exactly one authentication mechanism must be provided.",
            )
            raise RuntimeError("This line should never be reached.")

        if token_present:
            self._verify_token_for_run(token, run_id, context)
            return

        verify_superexec_signed_metadata(
            context=context,
            method=method,
            plugin_type=self._EXEC_PLUGIN_TYPE,
            cfg=self.superexec_auth_config,
        )
