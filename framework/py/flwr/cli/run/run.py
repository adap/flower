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
"""Flower command line interface `run` command."""


import hashlib
import json
import subprocess
from pathlib import Path
from typing import Annotated, Any

import click
import typer

from flwr.cli.build import build_fab_from_disk, get_fab_filename
from flwr.cli.config_migration import migrate, warn_if_federation_config_overrides
from flwr.cli.config_utils import load_and_validate
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE, RUN_CONFIG_HELP_MESSAGE
from flwr.cli.flower_config import (
    _serialize_simulation_options,
    read_superlink_connection,
)
from flwr.cli.typing import SuperLinkConnection, SuperLinkSimulationOptions
from flwr.common.config import (
    get_metadata_from_config,
    parse_config_args,
    user_config_to_configrecord,
)
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat
from flwr.common.serde import config_record_to_proto, fab_to_proto, user_config_to_proto
from flwr.common.typing import Fab
from flwr.proto.control_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.utils import parse_app_spec

from ..log import start_stream
from ..utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)

CONN_REFRESH_PERIOD = 60  # Connection refresh period for log streaming (seconds)


# pylint: disable-next=too-many-locals, too-many-branches, R0913, R0917
def run(
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower App to run."),
    ] = Path("."),
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    run_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--run-config",
            "-c",
            help=RUN_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
    federation_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
            hidden=True,
        ),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Use `--stream` with `flwr run` to display logs;\n "
            "logs are not streamed by default.",
        ),
    ] = False,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """Run Flower App."""
    with cli_output_handler(output_format=output_format) as is_json:
        # Warn `--federation-config` is ignored
        warn_if_federation_config_overrides(federation_config_overrides)

        # Migrate legacy usage if any
        migrate(str(app), [], ignore_legacy_usage=True)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)

        # Determine if app is remote
        app_spec = None
        config: dict[str, Any] = {}
        if (app_str := str(app)).startswith("@"):
            # Validate app version and ID format
            try:
                _ = parse_app_spec(app_str)
            except ValueError as e:
                raise click.ClickException(str(e)) from e

            app_spec = app_str

        # Validate TOML configuration for local app
        else:
            config, warnings = load_and_validate(app / FAB_CONFIG_FILE)
            if warnings:
                typer.secho(
                    "Missing recommended fields in Flower App configuration "
                    "(pyproject.toml):\n"
                    + "\n".join([f"- {line}" for line in warnings]),
                    fg=typer.colors.YELLOW,
                    bold=True,
                )

        if superlink_connection.address:
            _run_with_control_api(
                app,
                config,
                superlink_connection,
                run_config_overrides,
                stream,
                is_json,
                app_spec,
            )
        else:
            _run_without_control_api(
                app=app,
                simulation_options=superlink_connection.options,  # type: ignore
                config_overrides=run_config_overrides,
            )


# pylint: disable-next=R0913, R0914, R0917
def _run_with_control_api(
    app: Path,
    config: dict[str, Any],
    superlink_connection: SuperLinkConnection,
    config_overrides: list[str] | None,
    stream: bool,
    is_json: bool,
    app_spec: str | None,
) -> None:
    channel = None
    is_remote_app = app_spec is not None
    try:
        channel = init_channel_from_connection(superlink_connection)
        stub = ControlStub(channel)

        # Build FAB if local app
        if not is_remote_app:
            fab_bytes = build_fab_from_disk(app)
            fab_hash = hashlib.sha256(fab_bytes).hexdigest()
            fab_id, fab_version = get_metadata_from_config(config)
            fab = Fab(fab_hash, fab_bytes, {})
        # Skip FAB build if remote app
        else:
            # Use empty values for FAB
            fab_id = fab_version = fab_hash = ""
            fab = Fab(fab_hash, b"", {})

        # Construct a `ConfigRecord` out of a flattened `UserConfig`
        options = {}
        if superlink_connection.options:
            options = _serialize_simulation_options(superlink_connection.options)

        c_record = user_config_to_configrecord(options)

        req = StartRunRequest(
            fab=fab_to_proto(fab),
            override_config=user_config_to_proto(parse_config_args(config_overrides)),
            federation=superlink_connection.federation or "",
            federation_options=config_record_to_proto(c_record),
            app_spec=app_spec or "",
        )
        with flwr_cli_grpc_exc_handler():
            res = stub.StartRun(req)

        if res.HasField("run_id"):
            typer.secho(
                f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN
            )
        else:
            raise click.ClickException("Failed to start run")

        if is_json:
            # Only include FAB metadata if we actually built a local FAB
            payload: dict[str, Any] = {
                "success": res.HasField("run_id"),
                "run-id": f"{res.run_id}" if res.HasField("run_id") else None,
            }
            if not is_remote_app:
                payload.update(
                    {
                        "fab-id": fab_id,
                        "fab-name": fab_id.rsplit("/", maxsplit=1)[-1],
                        "fab-version": fab_version,
                        "fab-hash": fab_hash[:8],
                        "fab-filename": get_fab_filename(config, fab_hash),
                    }
                )
            print_json_to_stdout(payload)

        if stream:
            start_stream(res.run_id, channel, CONN_REFRESH_PERIOD)
    finally:
        if channel:
            channel.close()


def _run_without_control_api(
    app: Path | None,
    simulation_options: SuperLinkSimulationOptions,
    config_overrides: list[str] | None,
) -> None:

    num_supernodes = simulation_options.num_supernodes
    verbose = False  # bool | None = superlink_connection.options.verbose

    command = [
        "flower-simulation",
        "--app",
        f"{app}",
        "--num-supernodes",
        f"{num_supernodes}",
    ]

    if simulation_options.backend:
        # Stringify as JSON
        backend_serial = _serialize_simulation_options(simulation_options)
        command.extend(["--backend-config", json.dumps(backend_serial)])

    if verbose:
        command.extend(["--verbose"])

    if config_overrides:
        command.extend(["--run-config", f"{' '.join(config_overrides)}"])

    # Run the simulation
    subprocess.run(
        command,
        check=True,
        text=True,
    )
