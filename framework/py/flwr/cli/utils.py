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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import grpc
import pathspec
import requests
import typer

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    CREDENTIALS_DIR,
    FLWR_DIR,
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
from flwr.common.version import package_version as flwr_version
from flwr.supercore.constant import APP_ID_PATTERN, APP_VERSION_PATTERN

from .auth_plugin import CliAuthPlugin, get_cli_plugin_class
from .cli_account_auth_interceptor import CliAccountAuthInterceptor
from .config_utils import validate_certificate_in_federation_config


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


def sanitize_project_name(name: str) -> str:
    """Sanitize the given string to make it a valid Python project name.

    This function replaces spaces, dots, slashes, and underscores with dashes, removes
    any characters not allowed in Python project names, makes the string lowercase, and
    ensures it starts with a valid character.

    Parameters
    ----------
    name : str
        The project name to sanitize.

    Returns
    -------
    str
        The sanitized project name that is valid for Python projects.
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


def get_account_auth_config_path(root_dir: Path, federation: str) -> Path:
    """Return the path to the account auth config file.

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


def account_auth_enabled(federation_config: dict[str, Any]) -> bool:
    """Check if account authentication is enabled in the federation config.

    Parameters
    ----------
    federation_config : dict[str, Any]
        The federation configuration dictionary.

    Returns
    -------
    bool
        True if account authentication is enabled, False otherwise.
    """
    enabled: bool = federation_config.get("enable-user-auth", False)
    enabled |= federation_config.get("enable-account-auth", False)
    if "enable-user-auth" in federation_config:
        typer.secho(
            "`enable-user-auth` is deprecated and will be removed in a future "
            "release. Please use `enable-account-auth` instead.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    return enabled


def retrieve_authn_type(config_path: Path) -> str:
    """Retrieve the auth type from the config file or return NOOP if not found.

    Parameters
    ----------
    config_path : Path
        Path to the authentication configuration file.

    Returns
    -------
    str
        The authentication type string, or AuthnType.NOOP if not found.
    """
    try:
        with config_path.open("r", encoding="utf-8") as file:
            json_file = json.load(file)
        authn_type: str = json_file[AUTHN_TYPE_JSON_KEY]
        return authn_type
    except (FileNotFoundError, KeyError):
        return AuthnType.NOOP


def load_cli_auth_plugin(
    root_dir: Path,
    federation: str,
    federation_config: dict[str, Any],
    authn_type: str | None = None,
) -> CliAuthPlugin:
    """Load the CLI-side account auth plugin for the given authn type.

    Parameters
    ----------
    root_dir : Path
        Root directory of the Flower project.
    federation : str
        Name of the federation.
    federation_config : dict[str, Any]
        Federation configuration dictionary.
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
    # Find the path to the account auth config file
    config_path = get_account_auth_config_path(root_dir, federation)

    # Determine the auth type if not provided
    # Only `flwr login` command can provide `authn_type` explicitly, as it can query the
    # SuperLink for the auth type.
    if authn_type is None:
        authn_type = AuthnType.NOOP
        if account_auth_enabled(federation_config):
            authn_type = retrieve_authn_type(config_path)

    # Retrieve auth plugin class and instantiate it
    try:
        auth_plugin_class = get_cli_plugin_class(authn_type)
        return auth_plugin_class(config_path)
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
            )
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            if e.details() == NO_ACCOUNT_AUTH_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Account authentication is not enabled on this SuperLink.",
                    fg=typer.colors.RED,
                    bold=True,
                )
            elif e.details() == NO_ARTIFACT_PROVIDER_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ The SuperLink does not support `flwr pull` command.",
                    fg=typer.colors.RED,
                    bold=True,
                )
            else:
                typer.secho(
                    "❌ The SuperLink cannot process this request. Please verify that "
                    "you set the address to its Control API endpoint correctly in your "
                    "`pyproject.toml`, and ensure that the Flower versions used by "
                    "the CLI and SuperLink are compatible.",
                    fg=typer.colors.RED,
                    bold=True,
                )
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            typer.secho(
                "❌ Permission denied.",
                fg=typer.colors.RED,
                bold=True,
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
            )
            raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.NOT_FOUND:
            if e.details() == RUN_ID_NOT_FOUND_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Run ID not found.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(code=1) from None
            if e.details() == NODE_NOT_FOUND_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Node ID not found for this account.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(code=1) from None
        if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
            if e.details() == PULL_UNFINISHED_RUN_MESSAGE:  # pylint: disable=E1101
                typer.secho(
                    "❌ Run is not finished yet. Artifacts can only be pulled after "
                    "the run is finished. You can check the run status with `flwr ls`.",
                    fg=typer.colors.RED,
                    bold=True,
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
                )
                raise typer.Exit(code=1) from None
            if e.details() == PUBLIC_KEY_NOT_VALID:  # pylint: disable=E1101
                typer.secho(
                    "❌ The provided public key is invalid. Please provide a valid "
                    "NIST EC public key.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(code=1) from None

            # Log details from grpc error directly
            typer.secho(
                f"❌ {e.details()}",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1) from None
        raise


def request_download_link(
    app_id: str, app_version: str | None, in_url: str, out_url: str
) -> str:
    """Request a download link for the given app from the Flower platform API.

    Parameters
    ----------
    app_id : str
        The application identifier.
    app_version : str | None
        The application version, or None for latest.
    in_url : str
        The API endpoint URL.
    out_url : str
        The key name for the download URL in the response.

    Returns
    -------
    str
        The download URL for the application.

    Raises
    ------
    typer.Exit
        If connection fails, app not found, or API request fails.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "app_id": app_id,  # send raw string of app_id
        "app_version": app_version,
        "flwr_version": flwr_version,
    }
    try:
        resp = requests.post(in_url, headers=headers, data=json.dumps(body), timeout=20)
    except requests.RequestException as e:
        typer.secho(
            f"Unable to connect to Platform API: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from e

    if resp.status_code == 404:
        error_message = resp.json()["detail"]
        if isinstance(error_message, dict):
            available_app_versions = error_message["available_app_versions"]
            available_versions_str = (
                ", ".join(map(str, available_app_versions))
                if available_app_versions
                else "None"
            )
            typer.secho(
                f"{app_id}=={app_version} not found in Platform API. "
                f"Available app versions for {app_id}: {available_versions_str}",
                fg=typer.colors.RED,
                err=True,
            )
        else:
            typer.secho(
                f"{app_id} not found in Platform API.",
                fg=typer.colors.RED,
                err=True,
            )
        raise typer.Exit(code=1)

    if not resp.ok:
        typer.secho(
            f"Platform API request failed with "
            f"status {resp.status_code}. Details: {resp.text}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    data = resp.json()
    if out_url not in data:
        typer.secho(
            "Invalid response from Platform API",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    return str(data[out_url])


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


def parse_app_spec(app_spec: str) -> tuple[str, str | None]:
    """Parse app specification string into app ID and version.

    Parameters
    ----------
    app_spec : str
        The app specification string in the format '@account/app' or
        '@account/app==x.y.z' (digits only).

    Returns
    -------
    tuple[str, str | None]
        A tuple containing the app ID and optional version.

    Raises
    ------
    typer.Exit
        If the app specification format is invalid.
    """
    if "==" in app_spec:
        app_id, app_version = app_spec.split("==")

        # Validate app version format
        if not re.match(APP_VERSION_PATTERN, app_version):
            typer.secho(
                "❌ Invalid app version. Expected format: x.y.z (digits only).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        app_id = app_spec
        app_version = None

    # Validate app_id format
    if not re.match(APP_ID_PATTERN, app_id):
        typer.secho(
            "❌ Invalid remote app ID. Expected format: '@account/app'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    return app_id, app_version
