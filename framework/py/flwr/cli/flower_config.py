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
"""Flower command line interface configuration utils."""


from typing import Any, cast

import tomli
import typer

from flwr.cli.constant import (
    DEFAULT_FLOWER_CONFIG_TOML,
    DEFAULT_SIMULATION_BACKEND_NAME,
    FLOWER_CONFIG_FILE,
    SimulationBackendConfigTomlKey,
    SimulationClientResourcesTomlKey,
    SimulationInitArgsTomlKey,
    SuperLinkConnectionTomlKey,
    SuperLinkSimulationOptionsTomlKey,
)
from flwr.cli.typing import (
    SimulationBackendConfig,
    SimulationClientResources,
    SimulationInitArgs,
    SuperLinkConnection,
    SuperLinkSimulationOptions,
)
from flwr.supercore.utils import get_flwr_home


def _parse_simulation_options(options: dict[str, Any]) -> SuperLinkSimulationOptions:
    """Parse simulation options from a dictionary in a SuperLink connection."""
    num_supernodes = options.get(SuperLinkSimulationOptionsTomlKey.NUM_SUPERNODES)
    # Validation handled in SuperLinkSimulationOptions.__post_init__

    backend_dict = options.get(SuperLinkSimulationOptionsTomlKey.BACKEND)
    simulation_backend: SimulationBackendConfig | None = None

    if isinstance(backend_dict, dict):
        # Parse client resources
        client_resources_dict = backend_dict.get(
            SimulationBackendConfigTomlKey.CLIENT_RESOURCES
        )
        client_resources: SimulationClientResources | None = None
        if isinstance(client_resources_dict, dict):
            client_resources = SimulationClientResources(
                num_cpus=client_resources_dict.get(
                    SimulationClientResourcesTomlKey.NUM_CPUS
                ),
                num_gpus=client_resources_dict.get(
                    SimulationClientResourcesTomlKey.NUM_GPUS
                ),
            )

        # Parse init args
        init_args_dict = backend_dict.get(SimulationBackendConfigTomlKey.INIT_ARGS)
        init_args: SimulationInitArgs | None = None
        if isinstance(init_args_dict, dict):
            init_args = SimulationInitArgs(
                num_cpus=init_args_dict.get(SimulationInitArgsTomlKey.NUM_CPUS),
                num_gpus=init_args_dict.get(SimulationInitArgsTomlKey.NUM_GPUS),
                logging_level=init_args_dict.get(
                    SimulationInitArgsTomlKey.LOGGING_LEVEL
                ),
                log_to_drive=init_args_dict.get(SimulationInitArgsTomlKey.LOG_TO_DRIVE),
            )

        simulation_backend = SimulationBackendConfig(
            name=backend_dict.get(
                SimulationBackendConfigTomlKey.NAME, DEFAULT_SIMULATION_BACKEND_NAME
            ),
            client_resources=client_resources,
            init_args=init_args,
        )

    # Note: validation happens in SuperLinkSimulationOptions.__post_init__
    return SuperLinkSimulationOptions(
        num_supernodes=cast(int, num_supernodes),
        backend=simulation_backend,
    )


def init_flwr_config() -> None:
    """Initialize the Flower configuration file."""
    config_path = get_flwr_home() / FLOWER_CONFIG_FILE

    if not config_path.exists():
        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Write Flower config file
        config_path.write_text(DEFAULT_FLOWER_CONFIG_TOML, encoding="utf-8")

        typer.secho(
            f"\nFlower configuration not found. Created default configuration"
            f" at {config_path}\n",
        )


def parse_superlink_connection(
    conn_dict: dict[str, Any], name: str
) -> SuperLinkConnection:
    """Parse SuperLink connection configuration from a TOML dictionary.

    Parameters
    ----------
    conn_dict : dict[str, Any]
        The TOML configuration dictionary for the connection.
    name : str
        The name of the connection.

    Returns
    -------
    SuperLinkConnection
        The parsed SuperLink connection configuration.
    """
    simulation_options: SuperLinkSimulationOptions | None = None
    if SuperLinkConnectionTomlKey.OPTIONS in conn_dict:
        options = conn_dict[SuperLinkConnectionTomlKey.OPTIONS]
        if isinstance(options, dict):
            simulation_options = _parse_simulation_options(options)
        else:
            raise ValueError(
                f"Invalid value for key '{SuperLinkConnectionTomlKey.OPTIONS}': "
                f"expected dict, but got {type(options).__name__}."
            )

    # Build and return SuperLinkConnection
    return SuperLinkConnection(
        name=name,
        address=conn_dict.get(SuperLinkConnectionTomlKey.ADDRESS),
        root_certificates=conn_dict.get(SuperLinkConnectionTomlKey.ROOT_CERTIFICATES),
        insecure=conn_dict.get(SuperLinkConnectionTomlKey.INSECURE),
        enable_account_auth=conn_dict.get(
            SuperLinkConnectionTomlKey.ENABLE_ACCOUNT_AUTH
        ),
        federation=conn_dict.get(SuperLinkConnectionTomlKey.FEDERATION),
        options=simulation_options,
    )


def load_flower_config() -> dict[str, Any]:
    """Load the Flower configuration file."""
    # Initialize config if it doesn't exist
    init_flwr_config()
    config_path = get_flwr_home() / FLOWER_CONFIG_FILE
    # Load config
    with config_path.open("rb") as toml_file:
        try:
            return tomli.load(toml_file)
        except tomli.TOMLDecodeError as err:
            typer.secho(
                f"❌ Failed to load the Flower configuration file ({config_path}). "
                "Please ensure it is valid TOML.\n"
                f"Error: {err}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from err


def read_superlink_connection(
    connection_name: str | None = None,
) -> SuperLinkConnection | None:
    """Read a SuperLink connection from the Flower configuration file.

    Parameters
    ----------
    connection_name : str | None
        The name of the SuperLink connection to load. If None, the default connection
        will be loaded.

    Returns
    -------
    SuperLinkConnection | None
        The SuperLink connection, or None if the config file is missing or the
        requested connection (or default) cannot be found.

    Raises
    ------
    typer.Exit
        Raised if the configuration file is corrupted, or if the requested
        connection (or default) cannot be found.
    """
    config_path = get_flwr_home() / FLOWER_CONFIG_FILE
    if not config_path.exists():
        return None

    try:
        toml_dict = load_flower_config()

        superlink_config = toml_dict.get(SuperLinkConnectionTomlKey.SUPERLINK, {})

        # Load the default SuperLink connection when not provided
        if connection_name is None:
            connection_name = superlink_config.get(SuperLinkConnectionTomlKey.DEFAULT)

        # Exit when no connection name is available
        if connection_name is None:
            typer.secho(
                "❌ No SuperLink connection set. A SuperLink connection needs to be "
                "provided or one must be set as default in the Flower "
                f"configuration file ({config_path}). Specify a default SuperLink "
                "connection by adding: \n\n[superlink]\ndefault = 'connection_name'\n\n"
                f"to the Flower configuration file ({config_path}).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Try to find the connection with the given name
        if connection_name not in superlink_config:
            typer.secho(
                f"❌ SuperLink connection '{connection_name}' not found in the "
                f"Flower configuration file ({config_path}).",
                fg=typer.colors.RED,
                err=True,
            )
            # If default was used, show a specific error message
            if connection_name == superlink_config.get(
                SuperLinkConnectionTomlKey.DEFAULT
            ):
                typer.secho(
                    f"Please check that the default connection '{connection_name}' "
                    "is defined in the [superlink] section.",
                    fg=typer.colors.RED,
                    err=True,
                )
            raise typer.Exit(code=1)

        conn_dict = superlink_config[connection_name]
        return parse_superlink_connection(conn_dict, connection_name)

    except ValueError as err:
        typer.secho(
            f"❌ Failed to parse the Flower configuration file ({config_path}). "
            f"{err}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from err
    except Exception as err:
        typer.secho(
            f"❌ An unexpected error occurred while reading the Flower configuration "
            f"file ({config_path}). {err}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from err
