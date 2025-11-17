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
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console

from flwr.cli.build import build_fab_from_disk, get_fab_filename
from flwr.cli.config_utils import load as load_toml
from flwr.cli.config_utils import (
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE, RUN_CONFIG_HELP_MESSAGE
from flwr.common.config import (
    flatten_dict,
    get_metadata_from_config,
    parse_config_args,
    user_config_to_configrecord,
)
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.common.serde import config_record_to_proto, fab_to_proto, user_config_to_proto
from flwr.common.typing import Fab
from flwr.proto.control_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.constant import NOOP_FEDERATION

from ..log import start_stream
from ..utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin

CONN_REFRESH_PERIOD = 60  # Connection refresh period for log streaming (seconds)


# pylint: disable-next=too-many-locals, too-many-branches, R0913, R0917
def run(
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower App to run."),
    ] = Path("."),
    federation: Annotated[
        str | None,
        typer.Argument(help="Name of the federation to run the app on."),
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
    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()
    try:
        if suppress_output:
            redirect_output(captured_output)

        # Determine if app is remote
        app_id = None
        if (app_str := str(app)).startswith("@"):
            if not re.match(r"^@(?P<user>[^/]+)/(?P<app>[^/]+)$", app_str):
                raise typer.BadParameter(
                    "Invalid remote app ID. Expected format: '@user_name/app_name'."
                )
            app_id = app_str
        is_remote_app = app_id is not None

        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

        # Disable the validation for remote apps
        pyproject_path = app / "pyproject.toml" if not is_remote_app else None
        # `./pyproject.toml` will be loaded when `pyproject_path` is None
        config, errors, warnings = load_and_validate(
            pyproject_path, check_module=not is_remote_app
        )
        config = process_loaded_project_config(config, errors, warnings)

        federation, federation_config = validate_federation_in_project_config(
            federation, config, federation_config_overrides
        )

        if "address" in federation_config:
            _run_with_control_api(
                app,
                federation,
                federation_config,
                run_config_overrides,
                stream,
                output_format,
                app_id,
            )
        else:
            _run_without_control_api(
                app, federation_config, run_config_overrides, federation
            )
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


# pylint: disable-next=R0913, R0914, R0917
def _run_with_control_api(
    app: Path,
    federation: str,
    federation_config: dict[str, Any],
    config_overrides: list[str] | None,
    stream: bool,
    output_format: str,
    app_id: str | None,
) -> None:
    channel = None
    is_remote_app = app_id is not None
    try:
        auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
        channel = init_channel(app, federation_config, auth_plugin)
        stub = ControlStub(channel)

        # Build FAB if local app
        if not is_remote_app:
            fab_bytes = build_fab_from_disk(app)
            fab_hash = hashlib.sha256(fab_bytes).hexdigest()
            config = cast(dict[str, Any], load_toml(app / FAB_CONFIG_FILE))
            fab_id, fab_version = get_metadata_from_config(config)
            fab = Fab(fab_hash, fab_bytes, {})
        # Skip FAB build if remote app
        else:
            # Use empty values for FAB
            fab_id = fab_version = fab_hash = ""
            fab = Fab(fab_hash, b"", {})

        real_federation: str = federation_config.get("federation", NOOP_FEDERATION)

        # Construct a `ConfigRecord` out of a flattened `UserConfig`
        fed_config = flatten_dict(federation_config.get("options", {}))
        c_record = user_config_to_configrecord(fed_config)

        req = StartRunRequest(
            fab=fab_to_proto(fab),
            override_config=user_config_to_proto(parse_config_args(config_overrides)),
            federation=real_federation,
            federation_options=config_record_to_proto(c_record),
            app_id=app_id or "",
        )
        with flwr_cli_grpc_exc_handler():
            res = stub.StartRun(req)

        if res.HasField("run_id"):
            typer.secho(
                f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN
            )
        else:
            if is_remote_app:
                typer.secho(
                    "❌ Failed to start run. Please check that the provided "
                    "app identifier (@user_name/app_name) is correct.",
                    fg=typer.colors.RED,
                )
            else:
                typer.secho("❌ Failed to start run", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        if output_format == CliOutputFormat.JSON:
            # Only include FAB metadata if we actually built a local FAB
            payload: dict[str, Any] = {
                "success": res.HasField("run_id"),
                "run-id": res.run_id if res.HasField("run_id") else None,
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
            restore_output()
            Console().print_json(json.dumps(payload))

        if stream:
            start_stream(res.run_id, channel, CONN_REFRESH_PERIOD)
    finally:
        if channel:
            channel.close()


def _run_without_control_api(
    app: Path | None,
    federation_config: dict[str, Any],
    config_overrides: list[str] | None,
    federation: str,
) -> None:
    try:
        num_supernodes = federation_config["options"]["num-supernodes"]
        verbose: bool | None = federation_config["options"].get("verbose")
        backend_cfg = federation_config["options"].get("backend", {})
    except KeyError as err:
        typer.secho(
            "❌ The project's `pyproject.toml` needs to declare the number of"
            " SuperNodes in the simulation. To simulate 10 SuperNodes,"
            " use the following notation:\n\n"
            f"[tool.flwr.federations.{federation}]\n"
            "options.num-supernodes = 10\n",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    command = [
        "flower-simulation",
        "--app",
        f"{app}",
        "--num-supernodes",
        f"{num_supernodes}",
    ]

    if backend_cfg:
        # Stringify as JSON
        command.extend(["--backend-config", json.dumps(backend_cfg)])

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
