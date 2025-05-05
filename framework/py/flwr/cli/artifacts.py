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
"""Flower command line interface `artifacts` command."""


import json
from pathlib import Path
from typing import Annotated, Optional, Union

import tomli_w
import typer
from safetensors.numpy import save_file

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.common import ArrayRecord, MetricRecord, RecordSet
from flwr.common.constant import FAB_CONFIG_FILE
from flwr.common.logger import restore_output
from flwr.common.serde import context_from_proto
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetArtifactRequest,
    GetArtifactResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

from .utils import init_channel, try_obtain_cli_auth_plugin, unauthenticated_exc_handler


def artifacts(  # pylint: disable=too-many-locals, too-many-branches, R0913, R0917
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation"),
    ] = None,
    run_id: Annotated[
        Optional[int],
        typer.Option(
            "--run-id",
            help="Specific run ID to display",
        ),
    ] = None,
    save_dir: Annotated[
        Path,
        typer.Option(
            "--save-dir",
            help="Directory where fetched artifacts are stored",
        ),
    ] = Path("artifacts"),
    federation_config_overrides: Annotated[
        Optional[list[str]],
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
) -> None:
    """Download artifacts generated during a Run."""
    try:
        # Load and validate federation config
        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

        pyproject_path = app / FAB_CONFIG_FILE if app else None
        config, errors, warnings = load_and_validate(path=pyproject_path)
        config = process_loaded_project_config(config, errors, warnings)
        federation, federation_config = validate_federation_in_project_config(
            federation, config, federation_config_overrides
        )
        exit_if_no_address(federation_config, "ls")
        channel = None
        try:
            auth_plugin = try_obtain_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ExecStub(channel)

            # Fetch artifacts for the specified run
            if run_id is not None:
                typer.echo(f"🔍 Fetching artifacts for run ID {run_id}...")
                restore_output()
                _fetch_and_save_artifacts(stub, run_id, save_dir)

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
        typer.secho(
            f"{err}",
            fg=typer.colors.RED,
            bold=True,
        )


def _fetch_and_save_artifacts(
    stub: ExecStub,
    run_id: int,
    save_dir: Path,
) -> None:
    """Fetch artifacts from run and save to disk."""
    with unauthenticated_exc_handler():
        res: GetArtifactResponse = stub.GetArtifacts(GetArtifactRequest(run_id=run_id))
    if not res.context:
        raise ValueError(f"Run ID {run_id} not found")

    # deserialize
    context = context_from_proto(res.context)

    # Create save dir
    save_dir = save_dir / str(run_id)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    with (save_dir / "run_config.toml").open("wb") as f:
        f.write(tomli_w.dumps(context.run_config).encode("utf-8"))

    # Save logs
    _save_logs_to_file(log=res.log, file_name=save_dir / "logs.txt")

    # Save MetricRecords
    _metric_records_to_json(context.state, save_dir)

    # Save ArrayRecords
    _array_records_to_safetensors(context.state, save_dir)


def _save_logs_to_file(log: str, file_name: str):
    """Save logs to file."""
    with open(str(file_name), "w", encoding="utf-8") as file:
        file.write(log)


def _metric_records_to_json(record_set: RecordSet, save_dir: Path):
    """Save all `MetricRecord` as a single JSON."""
    path = save_dir / "metric_records.json"
    serializable_dict: dict[str, dict[str, Union[float, int]]] = {}
    for record_name, record in record_set.items():
        if isinstance(record, MetricRecord):
            serializable_dict[record_name] = {k: v for k, v in record.items()}
    path.write_text(json.dumps(serializable_dict, indent=2), encoding="utf-8")


def _array_records_to_safetensors(record_set: RecordSet, save_dir: Path):
    """Save each `ArrayRecord` as a safetensor file."""
    for record_name, record in record_set.items():
        if isinstance(record, ArrayRecord):
            # Express record as dict of ndarrays
            save_file(
                {k: v.numpy() for k, v in record.items()},
                save_dir / f"{record_name}.safetensors",
            )
