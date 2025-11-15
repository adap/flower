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
"""Flower command line interface `supernode register` command."""


import io
import json
from pathlib import Path
from typing import Annotated

import typer
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from rich.console import Console

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    RegisterNodeRequest,
    RegisterNodeResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.primitives.asymmetric import public_key_to_bytes, uses_nist_ec_curve

from ..utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def register(  # pylint: disable=R0914
    public_key: Annotated[
        Path,
        typer.Argument(
            help="Path to a P-384 (or any other NIST EC curve) public key file.",
        ),
    ],
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        str | None,
        typer.Argument(help="Name of the federation"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """Add a SuperNode to the federation."""
    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()

    # Load public key
    public_key_path = Path(public_key)
    public_key_bytes = try_load_public_key(public_key_path)

    try:
        if suppress_output:
            redirect_output(captured_output)

        # Load and validate federation config
        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

        pyproject_path = app / FAB_CONFIG_FILE if app else None
        config, errors, warnings = load_and_validate(path=pyproject_path)
        config = process_loaded_project_config(config, errors, warnings)
        federation, federation_config = validate_federation_in_project_config(
            federation, config
        )
        exit_if_no_address(federation_config, "supernode register")

        channel = None
        try:
            auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ControlStub(channel)  # pylint: disable=unused-variable # noqa: F841

            _register_node(
                stub=stub, public_key=public_key_bytes, output_format=output_format
            )

        except ValueError as err:
            typer.secho(
                f"❌ {err}",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1) from err
        finally:
            if channel:
                channel.close()

    except (typer.Exit, Exception) as err:  # pylint: disable=broad-except
        if suppress_output:
            restore_output()
            e_message = captured_output.getvalue()
            print_json_error(e_message, err)
        else:
            typer.secho(
                f"{err}",
                fg=typer.colors.RED,
                bold=True,
            )
    finally:
        if suppress_output:
            restore_output()
        captured_output.close()


def _register_node(stub: ControlStub, public_key: bytes, output_format: str) -> None:
    """Register a node."""
    with flwr_cli_grpc_exc_handler():
        response: RegisterNodeResponse = stub.RegisterNode(
            request=RegisterNodeRequest(public_key=public_key)
        )
    if response.node_id:
        typer.secho(
            f"✅ SuperNode {response.node_id} registered successfully.",
            fg=typer.colors.GREEN,
        )
        if output_format == CliOutputFormat.JSON:
            run_output = json.dumps(
                {
                    "success": True,
                    "node-id": response.node_id,
                }
            )
            restore_output()
            Console().print_json(run_output)
    else:
        typer.secho("❌ SuperNode couldn't be registered.", fg=typer.colors.RED)


def try_load_public_key(public_key_path: Path) -> bytes:
    """Try to load a public key from a file."""
    if not public_key_path.exists():
        typer.secho(
            f"❌ Public key file '{public_key_path}' does not exist.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    with open(public_key_path, "rb") as key_file:
        try:
            public_key = serialization.load_ssh_public_key(key_file.read())

            if not isinstance(public_key, ec.EllipticCurvePublicKey):
                raise ValueError(f"Not an EC public key, got {type(public_key)}")

            # Verify it's one of the approved NIST curves
            if not uses_nist_ec_curve(public_key):
                raise ValueError(
                    f"EC curve {public_key.curve.name} is not an approved NIST curve"
                )

        except (ValueError, UnsupportedAlgorithm) as err:
            flwr_exit(
                ExitCode.FLWRCLI_NODE_AUTH_PUBLIC_KEY_INVALID,
                str(err),
            )
    return public_key_to_bytes(public_key)
