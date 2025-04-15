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
"""Flower command line interface utils."""


import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import grpc
import typer

from flwr.cli.cli_user_auth_interceptor import CliUserAuthInterceptor
from flwr.common.auth_plugin import CliAuthPlugin
from flwr.common.constant import AUTH_TYPE_JSON_KEY, CREDENTIALS_DIR, FLWR_DIR
from flwr.common.grpc import (
    GRPC_MAX_MESSAGE_LENGTH,
    create_channel,
    on_channel_state_change,
)

from .auth_plugin import get_cli_auth_plugins
from .config_utils import validate_certificate_in_federation_config


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


def get_sha256_hash(file_path_or_int: Union[Path, int]) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    if isinstance(file_path_or_int, Path):
        with open(file_path_or_int, "rb") as f:
            while True:
                data = f.read(65536)  # Read in 64kB blocks
                if not data:
                    break
                sha256.update(data)
    elif isinstance(file_path_or_int, int):
        sha256.update(str(file_path_or_int).encode())
    return sha256.hexdigest()


def get_user_auth_config_path(root_dir: Path, federation: str) -> Path:
    """Return the path to the user auth config file.

    Additionally, a `.gitignore` file will be created in the Flower directory to
    include the `.credentials` folder to be excluded from git. If the `.gitignore`
    file already exists, a warning will be displayed if the `.credentials` entry is
    not found.
    """
    # Locate the credentials directory
    abs_flwr_dir = root_dir.absolute() / FLWR_DIR
    credentials_dir = abs_flwr_dir / CREDENTIALS_DIR
    credentials_dir.mkdir(parents=True, exist_ok=True)

    # Determine the absolute path of the Flower directory for .gitignore
    gitignore_path = abs_flwr_dir / ".gitignore"
    credential_entry = CREDENTIALS_DIR

    try:
        if gitignore_path.exists():
            with open(gitignore_path, encoding="utf-8") as gitignore_file:
                lines = gitignore_file.read().splitlines()

            # Warn if .credentials is not already in .gitignore
            if credential_entry not in lines:
                typer.secho(
                    f"`.gitignore` exists, but `{credential_entry}` entry not found. "
                    "Consider adding it to your `.gitignore` to exclude Flower "
                    "credentials from git.",
                    fg=typer.colors.YELLOW,
                    bold=True,
                )
        else:
            typer.secho(
                f"Creating a new `.gitignore` with `{credential_entry}` entry...",
                fg=typer.colors.BLUE,
            )
            # Create a new .gitignore with .credentials
            with open(gitignore_path, "w", encoding="utf-8") as gitignore_file:
                gitignore_file.write(f"{credential_entry}\n")
    except Exception as err:
        typer.secho(
            "❌ An error occurred while handling `.gitignore.` "
            f"Please check the permissions of `{gitignore_path}` and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    return credentials_dir / f"{federation}.json"


def try_obtain_cli_auth_plugin(
    root_dir: Path,
    federation: str,
    federation_config: dict[str, Any],
    auth_type: Optional[str] = None,
) -> Optional[CliAuthPlugin]:
    """Load the CLI-side user auth plugin for the given auth type."""
    # Check if user auth is enabled
    if not federation_config.get("enable-user-auth", False):
        return None

    # Check if TLS is enabled. If not, raise an error
    if federation_config.get("root-certificates") is None:
        typer.secho(
            "❌ User authentication requires TLS to be enabled. "
            "Please provide 'root-certificates' in the federation"
            " configuration.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    config_path = get_user_auth_config_path(root_dir, federation)

    # Get the auth type from the config if not provided
    # auth_type will be None for all CLI commands except login
    if auth_type is None:
        try:
            with config_path.open("r", encoding="utf-8") as file:
                json_file = json.load(file)
            auth_type = json_file[AUTH_TYPE_JSON_KEY]
        except (FileNotFoundError, KeyError):
            typer.secho(
                "❌ Missing or invalid credentials for user authentication. "
                "Please run `flwr login` to authenticate.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1) from None

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
    insecure, root_certificates_bytes = validate_certificate_in_federation_config(
        app, federation_config
    )

    # Initialize the CLI-side user auth interceptor
    interceptors: list[grpc.UnaryUnaryClientInterceptor] = []
    if auth_plugin is not None:
        auth_plugin.load_tokens()
        interceptors.append(CliUserAuthInterceptor(auth_plugin))

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


@contextmanager
def unauthenticated_exc_handler() -> Iterator[None]:
    """Context manager to handle gRPC UNAUTHENTICATED errors.

    It catches grpc.RpcError exceptions with UNAUTHENTICATED status, informs the user,
    and exits the application. All other exceptions will be allowed to escape.
    """
    try:
        yield
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            typer.secho(
                "❌ Authentication failed. Please run `flwr login`"
                " to authenticate and try again.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            typer.secho(
                "❌ User authentication is not enabled on this SuperLink.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1) from None
        raise
