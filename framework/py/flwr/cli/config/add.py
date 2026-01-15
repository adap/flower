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
"""Flower command line interface `config add` command."""


from typing import Any

import questionary
import typer

from flwr.cli.constant import (
    DEFAULT_SIMULATION_BACKEND_NAME,
    SimulationBackendConfigTomlKey,
    SimulationClientResourcesTomlKey,
    SimulationInitArgsTomlKey,
    SuperLinkConnectionTomlKey,
    SuperLinkSimulationOptionsTomlKey,
)
from flwr.cli.flower_config import read_flower_config, write_flower_config
from flwr.cli.typing import (
    SimulationBackendConfig,
    SimulationClientResources,
    SimulationInitArgs,
    SuperLinkConnection,
    SuperLinkSimulationOptions,
)


def add() -> None:
    """Add a new SuperLink connection to the Flower configuration."""
    typer.secho("Step 1: Configure SuperLink Connection", bold=True)
    name = questionary.text("Name of the SuperLink connection:").ask()
    if not name:
        typer.secho("❌ Name is required.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    is_simulation = questionary.confirm(
        "Is this a simulation connection?", default=False
    ).ask()

    simulation_options: SuperLinkSimulationOptions | None = None
    address: str | None = None
    root_certificates: str | None = None
    insecure: bool | None = None
    enable_account_auth: bool | None = None
    federation: str | None = None

    if is_simulation:
        typer.secho("\nStep 2: Configure Simulation Options", bold=True)
        num_supernodes = questionary.text(
            "Number of SuperNodes:",
            validate=lambda text: text.isdigit() or "Please enter a valid integer",
        ).ask()

        backend_name = questionary.text(
            "Simulation backend name:", default=DEFAULT_SIMULATION_BACKEND_NAME
        ).ask()

        # Client Resources
        typer.secho("\nStep 3: Configure Client Resources (Optional)", bold=True)
        client_num_cpus = questionary.text(
            "Client num_cpus (optional, e.g. 1.0):",
            validate=lambda text: (
                True
                if not text
                else (
                    text.replace(".", "", 1).isdigit() or "Please enter a valid number"
                )
            ),
        ).ask()
        client_num_gpus = questionary.text(
            "Client num_gpus (optional, e.g. 0.5):",
            validate=lambda text: (
                True
                if not text
                else (
                    text.replace(".", "", 1).isdigit() or "Please enter a valid number"
                )
            ),
        ).ask()

        client_resources: SimulationClientResources | None = None
        if client_num_cpus or client_num_gpus:
            client_resources = SimulationClientResources(
                num_cpus=float(client_num_cpus) if client_num_cpus else None,
                num_gpus=float(client_num_gpus) if client_num_gpus else None,
            )

        # Init Args
        typer.secho("\nStep 4: Configure Init Args (Optional)", bold=True)
        init_num_cpus = questionary.text(
            "Init num_cpus (optional, int):",
            validate=lambda text: (
                True if not text else (text.isdigit() or "Please enter a valid integer")
            ),
        ).ask()
        init_num_gpus = questionary.text(
            "Init num_gpus (optional, int):",
            validate=lambda text: (
                True if not text else (text.isdigit() or "Please enter a valid integer")
            ),
        ).ask()
        logging_level = questionary.text("Logging level (optional):").ask()
        log_to_drive = questionary.confirm(
            "Log to drive? (optional)", default=False
        ).ask()

        # questionary.confirm resturns bool. So we can't distinguish between
        # user skipping and user selecting False.
        init_args = SimulationInitArgs(
            num_cpus=int(init_num_cpus) if init_num_cpus else None,
            num_gpus=int(init_num_gpus) if init_num_gpus else None,
            logging_level=logging_level if logging_level else None,
            log_to_drive=log_to_drive,
        )

        backend_config = SimulationBackendConfig(
            name=backend_name, client_resources=client_resources, init_args=init_args
        )

        simulation_options = SuperLinkSimulationOptions(
            num_supernodes=int(num_supernodes), backend=backend_config
        )

    else:
        # Not a simulation
        address = questionary.text("SuperLink address (e.g. 127.0.0.1:9093):").ask()
        if address:
            address = address.strip().strip('"').strip("'")
        root_certificates = questionary.path("Root certificates path (optional):").ask()
        # If empty string, treat as None
        if not root_certificates:
            root_certificates = None

        insecure = questionary.confirm("Insecure connection?", default=True).ask()
        enable_account_auth = questionary.confirm(
            "Enable account auth?", default=False
        ).ask()
        federation = questionary.text("Federation (optional):").ask()
        if not federation:
            federation = None

    # Validate data creation by creating the object
    try:
        connection = SuperLinkConnection(
            name=name,
            address=address if address else None,
            root_certificates=root_certificates,
            insecure=insecure,
            enable_account_auth=enable_account_auth,
            federation=federation,
            options=simulation_options,
        )
    except ValueError as err:
        typer.secho(f"❌ Invalid configuration: {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from err

    # Read config
    toml_dict, config_path = read_flower_config()

    # Check if connection exists
    if SuperLinkConnectionTomlKey.SUPERLINK not in toml_dict:
        toml_dict[SuperLinkConnectionTomlKey.SUPERLINK] = {}

    if name in toml_dict[SuperLinkConnectionTomlKey.SUPERLINK]:
        if not questionary.confirm(
            f"Connection '{name}' already exists. Overwrite?"
        ).ask():
            typer.secho("Operation cancelled.")
            raise typer.Exit()

    # Convert to Dict for TOML
    conn_dict: dict[str, Any] = {}
    if connection.address:
        conn_dict[SuperLinkConnectionTomlKey.ADDRESS] = connection.address
    if connection.root_certificates:
        conn_dict[SuperLinkConnectionTomlKey.ROOT_CERTIFICATES] = (
            connection.root_certificates
        )
    if connection.insecure is not None:
        conn_dict[SuperLinkConnectionTomlKey.INSECURE] = connection.insecure
    if connection.enable_account_auth is not None:
        conn_dict[SuperLinkConnectionTomlKey.ENABLE_ACCOUNT_AUTH] = (
            connection.enable_account_auth
        )
    if connection.federation:
        conn_dict[SuperLinkConnectionTomlKey.FEDERATION] = connection.federation

    if connection.options:
        sim_options_dict: dict[str, Any] = {
            SuperLinkSimulationOptionsTomlKey.NUM_SUPERNODES: (
                connection.options.num_supernodes
            )
        }
        if connection.options.backend:
            backend_dict: dict[str, Any] = {}
            if connection.options.backend.name:
                backend_dict[SimulationBackendConfigTomlKey.NAME] = (
                    connection.options.backend.name
                )

            if connection.options.backend.client_resources:
                cr = connection.options.backend.client_resources
                cr_dict: dict[str, Any] = {}
                if cr.num_cpus is not None:
                    cr_dict[SimulationClientResourcesTomlKey.NUM_CPUS] = cr.num_cpus
                if cr.num_gpus is not None:
                    cr_dict[SimulationClientResourcesTomlKey.NUM_GPUS] = cr.num_gpus
                if cr_dict:
                    backend_dict[SimulationBackendConfigTomlKey.CLIENT_RESOURCES] = (
                        cr_dict
                    )

            if connection.options.backend.init_args:
                ia = connection.options.backend.init_args
                ia_dict: dict[str, Any] = {}
                if ia.num_cpus is not None:
                    ia_dict[SimulationInitArgsTomlKey.NUM_CPUS] = ia.num_cpus
                if ia.num_gpus is not None:
                    ia_dict[SimulationInitArgsTomlKey.NUM_GPUS] = ia.num_gpus
                if ia.logging_level:
                    ia_dict[SimulationInitArgsTomlKey.LOGGING_LEVEL] = ia.logging_level
                if ia.log_to_drive is not None:
                    ia_dict[SimulationInitArgsTomlKey.LOG_TO_DRIVE] = ia.log_to_drive
                if ia_dict:
                    backend_dict[SimulationBackendConfigTomlKey.INIT_ARGS] = ia_dict

            if backend_dict:
                sim_options_dict[SuperLinkSimulationOptionsTomlKey.BACKEND] = (
                    backend_dict
                )

        conn_dict[SuperLinkConnectionTomlKey.OPTIONS] = sim_options_dict

    toml_dict[SuperLinkConnectionTomlKey.SUPERLINK][name] = conn_dict

    # Write config
    write_flower_config(toml_dict)

    typer.secho(
        f"✅ Successfully added SuperLink connection '{name}' to {config_path}",
        fg=typer.colors.GREEN,
    )
