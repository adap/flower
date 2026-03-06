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


from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from flwr.cli.utils import (
    cli_output_control_stub,
    flwr_cli_grpc_exc_handler,
    print_json_to_stdout,
)
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListInvitationsRequest,
    ListInvitationsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_pb2 import Invitation  # pylint: disable=E0611


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
    is_json: bool,
    verbose: bool,
) -> None:
    """Send a list invitations request."""
    with flwr_cli_grpc_exc_handler():
        response: ListInvitationsResponse = stub.ListInvitations(request)

    received_invitations = _filter_invitations(
        invitations=list(response.received_invitations),
        verbose=verbose,
    )
    created_invitations = _filter_invitations(
        invitations=list(response.created_invitations),
        verbose=verbose,
    )

    if is_json:
        print_json_to_stdout(
            _to_json(
                created_invitations=created_invitations,
                received_invitations=received_invitations,
            )
        )
    else:
        console = Console()
        console.print("Your invitations:")
        console.print(_to_received_invitations_table(received_invitations))
        console.print()
        console.print("Invitations you created:")
        console.print(_to_created_invitations_table(created_invitations))
        if not verbose and not received_invitations and not created_invitations:
            console.print(
                "[dim]No pending invitations. Use --verbose to show all statuses.[/dim]"
            )


def _filter_invitations(
    invitations: list[Invitation], verbose: bool
) -> list[Invitation]:
    """Filter and sort invitations for presentation."""
    filtered = (
        invitations
        if verbose
        else [
            invitation for invitation in invitations if _is_pending(invitation.status)
        ]
    )
    return sorted(
        filtered,
        key=lambda invitation: (
            invitation.federation_name.lower(),
            invitation.inviter.name.lower(),
            invitation.invitee.name.lower(),
            invitation.created_at,
        ),
    )


def _is_pending(status: str) -> bool:
    """Return True if invitation status is pending."""
    return status.strip().lower() == "pending"


def _to_received_invitations_table(invitations: list[Invitation]) -> Table:
    """Render invitations received by the current account."""
    table = Table(header_style="bold cyan", show_lines=True)
    table.add_column(
        Text("Federation Name", justify="center"),
        style="bright_black",
        no_wrap=True,
    )
    table.add_column(Text("Invited By", justify="center"), no_wrap=True)
    table.add_column(Text("Status", justify="center"), no_wrap=True)

    for invitation in invitations:
        status_label = _format_status(invitation.status)
        status_style = _status_style(invitation.status)
        table.add_row(
            invitation.federation_name,
            invitation.inviter.name,
            f"[{status_style}]{status_label}[/{status_style}]",
        )

    return table


def _to_created_invitations_table(invitations: list[Invitation]) -> Table:
    """Render invitations created by the current account."""
    table = Table(header_style="bold cyan", show_lines=True)
    table.add_column(
        Text("Federation Name", justify="center"),
        style="bright_black",
        no_wrap=True,
    )
    table.add_column(Text("Invited", justify="center"), no_wrap=True)
    table.add_column(Text("Status", justify="center"), no_wrap=True)

    for invitation in invitations:
        status_label = _format_status(invitation.status)
        status_style = _status_style(invitation.status)
        table.add_row(
            invitation.federation_name,
            invitation.invitee.name,
            f"[{status_style}]{status_label}[/{status_style}]",
        )

    return table


def _status_style(status: str) -> str:
    """Return rich style name for an invitation status."""
    normalized = status.strip().lower()
    if normalized == "accepted":
        return "green"
    if normalized == "pending":
        return "yellow"
    if normalized == "rejected":
        return "red"
    return "bright_black"


def _format_status(status: str) -> str:
    """Format status for readable CLI output."""
    normalized = status.strip().replace("_", " ")
    return " ".join(part.capitalize() for part in normalized.split()) or "Unknown"


def _to_json(
    created_invitations: list[Invitation],
    received_invitations: list[Invitation],
) -> dict[str, Any]:
    """Convert invitations to JSON serializable structure."""
    return {
        "success": True,
        "received-invitations": [
            {
                "federation-name": invitation.federation_name,
                "invited-by": invitation.inviter.name,
                "status": invitation.status,
                "created-at": invitation.created_at,
                "status-changed-at": invitation.status_changed_at,
            }
            for invitation in received_invitations
        ],
        "created-invitations": [
            {
                "federation-name": invitation.federation_name,
                "invited": invitation.invitee.name,
                "status": invitation.status,
                "created-at": invitation.created_at,
                "status-changed-at": invitation.status_changed_at,
            }
            for invitation in created_invitations
        ],
    }
