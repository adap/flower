# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface utils."""

import hashlib
import json
import re
from logging import DEBUG
from pathlib import Path
from typing import Any, Callable, Optional, cast

import grpc
import typer

from flwr.cli.cli_user_auth_interceptor import CliUserAuthInterceptor
from flwr.common.address import parse_address
from flwr.common.auth_plugin import CliAuthPlugin
from flwr.common.constant import AUTH_TYPE, CREDENTIALS_DIR, FLWR_DIR
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log

from .config_utils import validate_certificate_in_federation_config

try:
    from flwr.ee import get_cli_auth_plugins
except ImportError:

    def get_cli_auth_plugins() -> dict[str, type[CliAuthPlugin]]:
        """Return all CLI authentication plugins."""
        raise NotImplementedError("No authentication plugins are currently supported.")


def prompt_text(
    text: str,
    predicate: Callable[[str], bool] = lambda _: True,
    default: Optional[str] = None,
) -> str:
    """Ask user to enter text input."""
    while True:
        result = typer.prompt(
            typer.style(f"\n💬 {text}", fg=typer.colors.MAGENTA, bold=True),
            default=default,
        )
        if predicate(result) and len(result) > 0:
            break
        print(typer.style("❌ Invalid entry", fg=typer.colors.RED, bold=True))

    return cast(str, result)


def prompt_options(text: str, options: list[str]) -> str:
    """Ask user to select one of the given options and return the selected item."""
    # Turn options into a list with index as in " [ 0] quickstart-pytorch"
    options_formatted = [
        " [ "
        + typer.style(index, fg=typer.colors.GREEN, bold=True)
        + "]"
        + f" {typer.style(name, fg=typer.colors.WHITE, bold=True)}"
        for index, name in enumerate(options)
    ]

    while True:
        index = typer.prompt(
            "\n"
            + typer.style(f"💬 {text}", fg=typer.colors.MAGENTA, bold=True)
            + "\n\n"
            + "\n".join(options_formatted)
            + "\n\n\n"
        )
        try:
            options[int(index)]  # pylint: disable=expression-not-assigned
            break
        except IndexError:
            print(typer.style("❌ Index out of range", fg=typer.colors.RED, bold=True))
            continue
        except ValueError:
            print(
                typer.style("❌ Please choose a number", fg=typer.colors.RED, bold=True)
            )
            continue

    result = options[int(index)]
    return result


def is_valid_project_name(name: str) -> bool:
    """Check if the given string is a valid Python project name.

    A valid project name must start with a letter and can only contain letters, digits,
    and hyphens.
    """
    if not name:
        return False

    # Check if the first character is a letter
    if not name[0].isalpha():
        return False

    # Check if the rest of the characters are valid (letter, digit, or dash)
    for char in name[1:]:
        if not (char.isalnum() or char in "-"):
            return False

    return True


def sanitize_project_name(name: str) -> str:
    """Sanitize the given string to make it a valid Python project name.

    This version replaces spaces, dots, slashes, and underscores with dashes, removes
    any characters not allowed in Python project names, makes the string lowercase, and
    ensures it starts with a valid character.
    """
    # Replace whitespace with '_'
    name_with_hyphens = re.sub(r"[ ./_]", "-", name)

    # Allowed characters in a module name: letters, digits, underscore
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    )

    # Make the string lowercase
    sanitized_name = name_with_hyphens.lower()

    # Remove any characters not allowed in Python module names
    sanitized_name = "".join(c for c in sanitized_name if c in allowed_chars)

    # Ensure the first character is a letter or underscore
    while sanitized_name and (
        sanitized_name[0].isdigit() or sanitized_name[0] not in allowed_chars
    ):
        sanitized_name = sanitized_name[1:]

    return sanitized_name


def get_sha256_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)  # Read in 64kB blocks
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def get_user_auth_config_path(
    root_dir: Path, federation: str, server_address: str
) -> Path:
    """Return the path to the user auth config file."""
    # Parse the server address
    parsed_addr = parse_address(server_address)
    if parsed_addr is None:
        raise ValueError(f"Invalid server address: {server_address}")
    host, port, is_v6 = parsed_addr
    formatted_addr = f"[{host}]_{port}" if is_v6 else f"{host}_{port}"

    # Locate the credentials directory
    credentials_dir = root_dir.absolute() / FLWR_DIR / CREDENTIALS_DIR
    credentials_dir.mkdir(parents=True, exist_ok=True)
    return credentials_dir / f"{federation}_{formatted_addr}.json"


def try_obtain_cli_auth_plugin(
    root_dir: Path,
    federation: str,
    federation_config: dict[str, Any],
    auth_type: Optional[str] = None,
) -> Optional[CliAuthPlugin]:
    """Load the CLI-side user auth plugin for the given auth type."""
    config_path = get_user_auth_config_path(
        root_dir, federation, federation_config["address"]
    )

    # Load the config file if it exists
    config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    # This is the case when the user auth is not enabled
    elif auth_type is None:
        return None

    # Get the auth type from the config if not provided
    if auth_type is None:
        if AUTH_TYPE not in config:
            return None
        auth_type = config[AUTH_TYPE]

    # Retrieve auth plugin class and instantiate it
    try:
        all_plugins: dict[str, type[CliAuthPlugin]] = get_cli_auth_plugins()
        auth_plugin_class = all_plugins[auth_type]
        return auth_plugin_class(config_path)
    except KeyError:
        typer.echo(f"❌ Unknown user authentication type: {auth_type}")
        raise typer.Exit(code=1) from None
    except ImportError:
        typer.echo("❌ No authentication plugins are currently supported.")
        raise typer.Exit(code=1) from None


def init_channel(
    app: Path, federation_config: dict[str, Any], auth_plugin: Optional[CliAuthPlugin]
) -> grpc.Channel:
    """Initialize gRPC channel to the Exec API."""

    def on_channel_state_change(channel_connectivity: str) -> None:
        """Log channel connectivity."""
        log(DEBUG, channel_connectivity)

    insecure, root_certificates_bytes = validate_certificate_in_federation_config(
        app, federation_config
    )

    # Initialize the CLI-side user auth interceptor
    interceptors: list[grpc.UnaryUnaryClientInterceptor] = []
    if auth_plugin is not None:
        auth_plugin.load_tokens()
        interceptors = CliUserAuthInterceptor(auth_plugin)

    # Create the gRPC channel
    channel = create_channel(
        server_address=federation_config["address"],
        insecure=insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=interceptors or None,
    )
    channel.subscribe(on_channel_state_change)
    return channel
