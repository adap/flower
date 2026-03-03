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
"""Flower command line interface `federation invite list` command."""


from typing import Annotated

import typer

from flwr.cli.utils import cli_output_control_stub, flwr_cli_grpc_exc_handler
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListInvitationsRequest,
    ListInvitationsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub


def ls(
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show non-pending invitations.",
        ),
    ] = False,
) -> None:
    """List invitations addressed to you and invitations created by you (alias: ls)."""
    with cli_output_control_stub(superlink, output_format) as (stub, is_json):
        request = ListInvitationsRequest()
        _list_invitations(
            stub=stub,
            request=request,
            is_json=is_json,
            verbose=verbose,
        )


def _list_invitations(
    stub: ControlStub,
    request: ListInvitationsRequest,
    is_json: bool,  # pylint: disable=W0613
    verbose: bool,  # pylint: disable=W0613
) -> None:
    """Send a list invitations request."""
    with flwr_cli_grpc_exc_handler():
        _: ListInvitationsResponse = stub.ListInvitations(request)

    raise NotImplementedError
