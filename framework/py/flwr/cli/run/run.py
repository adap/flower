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
from typing import Annotated, Any, Optional, cast

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

from ..log import start_stream
from ..utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin

CONN_REFRESH_PERIOD = 60  # Connection refresh period for log streaming (seconds)


# pylint: disable-next=too-many-locals, R0913, R0917
def run(
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower App to run."),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation to run the app on."),
    ] = None,
    run_config_overrides: Annotated[
        Optional[list[str]],
        typer.Option(
            "--run-config",
            "-c",
            help=RUN_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
    federation_config_overrides: Annotated[
        Optional[list[str]],
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

        original_app_str = str(app) if app is not None else ""
        remote_app_ref: Optional[str] = None  # "user_name/app_name" if given with "@"
        app_path: Path = app

        if original_app_str.startswith("@"):
            m = re.match(r"^@(?P<user>[^/]+)/(?P<app>[^/]+)$", original_app_str)
            if not m:
                raise typer.BadParameter(
                    "Invalid remote app ID. Expected format: '@user_name/app_name'."
                )
            app_name = m.group("app")
            user_name = m.group("user")

            # Use local folder named {app_name} for pyproject lookup
            # and downstream calls
            app_path = Path(app_name)
            remote_app_ref = f"{user_name}/{app_name}"

        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

        # Disable the validation due to the local empty project
        if remote_app_ref:
            config = load_toml(app_path / "pyproject.toml")
        else:
            pyproject_path = app_path / "pyproject.toml" if app_path else None
            config, errors, warnings = load_and_validate(path=pyproject_path)
            config = process_loaded_project_config(config, errors, warnings)

        federation, federation_config = validate_federation_in_project_config(
            federation, config, federation_config_overrides  # type: ignore[arg-type]
        )

        if "address" in federation_config:
            _run_with_control_api(
                app_path,
                federation,
                federation_config,
                run_config_overrides,
                stream,
                output_format,
                original_app_str,
                remote_app_ref,
            )
        else:
            _run_without_control_api(
                app_path, federation_config, run_config_overrides, federation
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
    config_overrides: Optional[list[str]],
    stream: bool,
    output_format: str,
    original_app_str: str,
    remote_app_ref: Optional[str] = None,
) -> None:
    channel = None
    try:
        auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
        channel = init_channel(app, federation_config, auth_plugin)
        stub = ControlStub(channel)

        # Build fab only if not a remote reference
        fab_id = fab_version = fab_hash = None
        if remote_app_ref:
            # Skip build; send a placeholder Fab containing the remote reference
            fab = Fab("", b"", {})
        else:
            fab_bytes = build_fab_from_disk(app)
            fab_hash = hashlib.sha256(fab_bytes).hexdigest()
            config = cast(dict[str, Any], load_toml(app / FAB_CONFIG_FILE))
            fab_id, fab_version = get_metadata_from_config(config)
            fab = Fab(fab_hash, fab_bytes, {})

        # Construct a `ConfigRecord` out of a flattened `UserConfig`
        fed_config = flatten_dict(federation_config.get("options", {}))
        c_record = user_config_to_configrecord(fed_config)

        req = StartRunRequest(
            fab=fab_to_proto(fab),
            override_config=user_config_to_proto(parse_config_args(config_overrides)),
            federation_options=config_record_to_proto(c_record),
            app_id=original_app_str,
        )
        with flwr_cli_grpc_exc_handler():
            res = stub.StartRun(req)

        if res.HasField("run_id"):
            typer.secho(
                f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN
            )
        else:
            if remote_app_ref:
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
            if not remote_app_ref:
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
    app: Optional[Path],
    federation_config: dict[str, Any],
    config_overrides: Optional[list[str]],
    federation: str,
) -> None:
    try:
        num_supernodes = federation_config["options"]["num-supernodes"]
        verbose: Optional[bool] = federation_config["options"].get("verbose")
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
