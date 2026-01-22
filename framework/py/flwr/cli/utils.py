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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import click
import grpc
import pathspec
import typer

from flwr.cli.typing import SuperLinkConnection
from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    NO_ACCOUNT_AUTH_MESSAGE,
    NO_ARTIFACT_PROVIDER_MESSAGE,
    NODE_NOT_FOUND_MESSAGE,
    PUBLIC_KEY_ALREADY_IN_USE_MESSAGE,
    PUBLIC_KEY_NOT_VALID,
    PULL_UNFINISHED_RUN_MESSAGE,
    REFRESH_TOKEN_KEY,
    RUN_ID_NOT_FOUND_MESSAGE,
    AuthnType,
)
from flwr.common.grpc import (
    GRPC_MAX_MESSAGE_LENGTH,
    create_channel,
    on_channel_state_change,
)
from flwr.supercore.credential_store import get_credential_store

from .auth_plugin import CliAuthPlugin, get_cli_plugin_class
from .cli_account_auth_interceptor import CliAccountAuthInterceptor
from .config_utils import (
    load_certificate_in_connection,
    validate_certificate_in_federation_config,
)
from .constant import AUTHN_TYPE_STORE_KEY


def prompt_text(
    text: str,
    predicate: Callable[[str], bool] = lambda _: True,
    default: str | None = None,
) -> str:
    """Ask user to enter text input.

    Parameters
    ----------
    text : str
        The prompt text to display to the user.
    predicate : Callable[[str], bool] (default: lambda _: True)
        A function to validate the user input. Default accepts all non-empty strings.
    default : str | None (default: None)
        Default value to use if user presses enter without input.

    Returns
    -------
    str
        The validated user input.
    """
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
    """Ask user to select one of the given options and return the selected item.

    Parameters
    ----------
    text : str
        The prompt text to display to the user.
    options : list[str]
        List of options to present to the user.

    Returns
    -------
    str
        The selected option from the list.
    """
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


def get_sha256_hash(file_path_or_int: Path | int) -> str:
    """Calculate the SHA-256 hash of a file or integer.

    Parameters
    ----------
    file_path_or_int : Path | int
        Either a path to a file to hash, or an integer to convert to string and hash.

    Returns
    -------
    str
        The SHA-256 hash as a hexadecimal string.
    """
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


def get_authn_type(host: str) -> str:
    """Retrieve the authentication type for the given host from the credential store.

    `AuthnType.NOOP` is returned if no authentication type is found.
    """
    store = get_credential_store()
    authn_type = store.get(AUTHN_TYPE_STORE_KEY % host)
    if authn_type is None:
        return AuthnType.NOOP
    return authn_type.decode("utf-8")


def load_cli_auth_plugin(
    root_dir: Path,
    federation: str,
    federation_config: dict[str, Any],
    authn_type: str | None = None,
) -> CliAuthPlugin:
    """."""
    raise RuntimeError(
        "Deprecated function. Use `load_cli_auth_plugin_from_connection`"
    )


def load_cli_auth_plugin_from_connection(
    host: str, authn_type: str | None = None
) -> CliAuthPlugin:
    """Load the CLI-side account auth plugin for the given connection.

    Parameters
    ----------
    host : str
        The SuperLink Control API address.
    authn_type : str | None
        Authentication type. If None, will be determined from config.

    Returns
    -------
    CliAuthPlugin
        The loaded authentication plugin instance.

    Raises
    ------
    typer.Exit
        If the authentication type is unknown.
    """
    # Determine the auth type if not provided
    # Only `flwr login` command can provide `authn_type` explicitly, as it can query the
    # SuperLink for the auth type.
    if authn_type is None:
        authn_type = get_authn_type(host)

    # Retrieve auth plugin class and instantiate it
    try:
        auth_plugin_class = get_cli_plugin_class(authn_type)
        return auth_plugin_class(host)
    except ValueError:
        typer.echo(f"❌ Unknown account authentication type: {authn_type}")
        raise typer.Exit(code=1) from None


def init_channel(
    app: Path, federation_config: dict[str, Any], auth_plugin: CliAuthPlugin
) -> grpc.Channel:
    """Initialize gRPC channel to the Control API.

    Parameters
    ----------
    app : Path
        Path to the Flower app directory.
    federation_config : dict[str, Any]
        Federation configuration dictionary containing address and TLS settings.
    auth_plugin : CliAuthPlugin
        Authentication plugin instance for handling credentials.

    Returns
    -------
    grpc.Channel
        Configured gRPC channel with authentication interceptors.
    """
    insecure, root_certificates_bytes = validate_certificate_in_federation_config(
        app, federation_config
    )

    # Load tokens
    auth_plugin.load_tokens()

    # Create the gRPC channel
    channel = create_channel(
        server_address=federation_config["address"],
        insecure=insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=[CliAccountAuthInterceptor(auth_plugin)],
    )
    channel.subscribe(on_channel_state_change)
    return channel


def require_superlink_address(connection: SuperLinkConnection) -> str:
    """Return the SuperLink address or exit if it is not configured."""
    if connection.address is None:
        cmd = click.get_current_context().command.name
        typer.secho(
            f"❌ `flwr {cmd}` currently works with a SuperLink. Ensure that the "
            "correct SuperLink (Control API) address is provided in `pyproject.toml`.",
            fg=typer.colors.RED,
            bold=True,
            err=True,
        )
        raise typer.Exit(code=1)


def init_channel_from_connection(
    connection: SuperLinkConnection, auth_plugin: CliAuthPlugin | None = None
) -> grpc.Channel:
    """Initialize gRPC channel to the Control API.

    Parameters
    ----------
    connection : SuperLinkConnection
        SuperLink connection configuration.
    auth_plugin : CliAuthPlugin | None (default: None)
        Authentication plugin instance for handling credentials.

    Returns
    -------
    grpc.Channel
        Configured gRPC channel with authentication interceptors.
    """
    ensure_connection_has_address(connection)
    address = cast(str, connection.address)

    root_certificates_bytes = load_certificate_in_connection(connection)

    # Load authentication plugin
    if auth_plugin is None:
        auth_plugin = load_cli_auth_plugin_from_connection(address)
    # Load tokens
    auth_plugin.load_tokens()

    # Create the gRPC channel
    channel = create_channel(
        server_address=address,
        insecure=connection.insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=[CliAccountAuthInterceptor(auth_plugin)],
    )
    channel.subscribe(on_channel_state_change)
    return channel


@contextmanager
def flwr_cli_grpc_exc_handler() -> Iterator[None]:  # pylint: disable=too-many-branches
    """Context manager to handle specific gRPC errors.

    Catches grpc.RpcError exceptions with UNAUTHENTICATED, UNIMPLEMENTED,
    UNAVAILABLE, PERMISSION_DENIED, NOT_FOUND, and FAILED_PRECONDITION statuses,
    informs the user, and exits the application. All other exceptions will be
    allowed to escape.

    Yields
    ------
    None
        Context manager yields nothing.

    Raises
    ------
    typer.Exit
        On handled gRPC error statuses with appropriate exit code.
    grpc.RpcError
        For unhandled gRPC error statuses.
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
                err=True,
            )
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            if e.details() == NO_ACCOUNT_AUTH_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Account authentication is not enabled on this SuperLink.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
            elif e.details() == NO_ARTIFACT_PROVIDER_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ The SuperLink does not support `flwr pull` command.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
            else:
                typer.secho(
                    "❌ The SuperLink cannot process this request. Please verify that "
                    "you set the address to its Control API endpoint correctly in your "
                    "`pyproject.toml`, and ensure that the Flower versions used by "
                    "the CLI and SuperLink are compatible.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            typer.secho(
                "❌ Permission denied.",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
            # pylint: disable-next=E1101
            typer.secho(e.details(), fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            typer.secho(
                "Connection to the SuperLink is unavailable. Please check your network "
                "connection and 'address' in the federation configuration.",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.NOT_FOUND:
            if e.details() == RUN_ID_NOT_FOUND_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Run ID not found.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
                raise typer.Exit(code=1) from None
            if e.details() == NODE_NOT_FOUND_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Node ID not found for this account.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
                raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
            if e.details() == PULL_UNFINISHED_RUN_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Run is not finished yet. Artifacts can only be pulled after "
                    "the run is finished. You can check the run status with `flwr ls`.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
                raise typer.Exit(code=1) from None
            if (
                e.details() == PUBLIC_KEY_ALREADY_IN_USE_MESSAGE
            ):  # pylint: disable=E1101
                typer.secho(
                    "❌ The provided public key is already in use by another "
                    "SuperNode.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
                raise typer.Exit(code=1) from None
            if e.details() == PUBLIC_KEY_NOT_VALID:  # pylint: disable=E1101
                typer.secho(
                    "❌ The provided public key is invalid. Please provide a valid "
                    "NIST EC public key.",
                    fg=typer.colors.RED,
                    bold=True,
                    err=True,
                )
                raise typer.Exit(code=1) from None

            # Log details from grpc error directly
            typer.secho(
                f"❌ {e.details()}",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
            raise typer.Exit(code=1) from None
        raise


def build_pathspec(patterns: Iterable[str]) -> pathspec.PathSpec:
    """Build a PathSpec from a list of GitIgnore-style patterns.

    Parameters
    ----------
    patterns : Iterable[str]
        Iterable of GitIgnore-style pattern strings.

    Returns
    -------
    pathspec.PathSpec
        Compiled PathSpec object for pattern matching.
    """
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def load_gitignore_patterns(file: Path | bytes) -> list[str]:
    """Load gitignore patterns from .gitignore file bytes.

    Parameters
    ----------
    file : Path | bytes
        The path to a .gitignore file or its bytes content.

    Returns
    -------
    list[str]
        List of gitignore patterns.
        Returns empty list if content can't be decoded or the file does not exist.
    """
    try:
        if isinstance(file, Path):
            content = file.read_text(encoding="utf-8")
        else:
            content = file.decode("utf-8")
        patterns = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return patterns
    except (UnicodeDecodeError, OSError):
        return []


def validate_credentials_content(creds_path: Path) -> str:
    """Load and validate the credentials file content.

    Ensures required keys exist:
      - AUTHN_TYPE_JSON_KEY
      - ACCESS_TOKEN_KEY
      - REFRESH_TOKEN_KEY
    """
    try:
        creds: dict[str, str] = json.loads(creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        typer.secho(
            f"Invalid credentials file at '{creds_path}': {err}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from err

    required_keys = [AUTHN_TYPE_JSON_KEY, ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY]
    missing = [key for key in required_keys if key not in creds]

    if missing:
        typer.secho(
            f"Credentials file '{creds_path}' is missing "
            f"required key(s): {', '.join(missing)}. Please log in again.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    return creds[ACCESS_TOKEN_KEY]
