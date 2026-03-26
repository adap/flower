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
"""Test for Flower command line interface utils."""


import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import click
import grpc
import pytest

from flwr.cli.typing import SuperLinkConnection, SuperLinkSimulationOptions
from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    FLWR_DIR,
    REFRESH_TOKEN_KEY,
)
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH

from .utils import (
    build_pathspec,
    collect_project_files,
    depth_of,
    flwr_cli_grpc_exc_handler,
    get_executed_command,
    get_sha256_hash,
    init_channel_from_connection,
    load_gitignore_patterns,
    optional_min_callback,
    validate_credentials_content,
)


class TestGetSHA256Hash(unittest.TestCase):
    """Unit tests for `get_sha256_hash` function."""

    def test_hash_with_integer(self) -> None:
        """Test the SHA-256 hash calculation when input is an integer."""
        # Prepare
        test_int = 13413
        expected_hash = hashlib.sha256(str(test_int).encode()).hexdigest()

        # Execute
        result = get_sha256_hash(test_int)

        # Assert
        self.assertEqual(result, expected_hash)

    def test_hash_with_file(self) -> None:
        """Test the SHA-256 hash calculation when input is a file path."""
        # Prepare - Create a temporary file with known content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test content for SHA-256 hashing.")
            temp_file_path = Path(temp_file.name)

        try:
            # Execute
            sha256 = hashlib.sha256()
            with open(temp_file_path, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    sha256.update(data)
            expected_hash = sha256.hexdigest()

            result = get_sha256_hash(temp_file_path)

            # Assert
            self.assertEqual(result, expected_hash)
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

    def test_empty_file(self) -> None:
        """Test the SHA-256 hash calculation for an empty file."""
        # Prepare
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)

        try:
            # Execute
            expected_hash = hashlib.sha256(b"").hexdigest()
            result = get_sha256_hash(temp_file_path)

            # Assert
            self.assertEqual(result, expected_hash)
        finally:
            os.remove(temp_file_path)

    def test_large_file(self) -> None:
        """Test the SHA-256 hash calculation for a large file."""
        # Prepare - Generate large data (e.g., 10 MB)
        large_data = b"a" * (10 * 1024 * 1024)  # 10 MB of 'a's
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(large_data)
            temp_file_path = Path(temp_file.name)

        try:
            expected_hash = hashlib.sha256(large_data).hexdigest()
            # Execute
            result = get_sha256_hash(temp_file_path)

            # Assert
            self.assertEqual(result, expected_hash)
        finally:
            os.remove(temp_file_path)

    def test_nonexistent_file(self) -> None:
        """Test the SHA-256 hash calculation when the file does not exist."""
        # Prepare
        nonexistent_path = Path("/path/to/nonexistent/file.txt")

        # Execute & assert
        with self.assertRaises(FileNotFoundError):
            get_sha256_hash(nonexistent_path)


def test_validate_credentials_content_success(tmp_path: Path) -> None:
    """Test the credentials content loading."""
    creds = {
        AUTHN_TYPE_JSON_KEY: "userpass",
        ACCESS_TOKEN_KEY: "abc",
        REFRESH_TOKEN_KEY: "def",
    }
    path = tmp_path / "creds.json"
    path.write_text(json.dumps(creds), encoding="utf-8")
    token = validate_credentials_content(path)
    assert token == "abc"


def test_load_gitignore_patterns(tmp_path: Path) -> None:
    """Test gitignore pattern loading."""
    path = tmp_path / ".gitignore"
    path.write_text("*.log\nsecret/\n# comment\n\n", encoding="utf-8")
    patterns_from_path = load_gitignore_patterns(path)
    patterns_from_bytes = load_gitignore_patterns(path.read_bytes())

    assert patterns_from_path == ["*.log", "secret/"]
    assert patterns_from_bytes == ["*.log", "secret/"]


def test_load_gitignore_patterns_with_pathspec() -> None:
    """Test gitignore patterns with pathspec matching."""
    patterns = load_gitignore_patterns(b"*.tmp\n")
    spec = build_pathspec(patterns + [f"{FLWR_DIR}/"])

    # Should match .tmp files
    assert spec.match_file("a.tmp") is True

    # Should match FLWR_DIR
    assert spec.match_file(f"{FLWR_DIR}/creds.json") is True

    # Should not match normal files
    assert spec.match_file("good.py") is False


def test_get_executed_command_single() -> None:
    """Test get_executed_command with a two-word command (e.g., flwr ls)."""
    root_group = click.Group("flwr")
    ls_cmd = click.Command("ls")

    with click.Context(root_group, info_name="flwr") as root_ctx:
        with click.Context(ls_cmd, parent=root_ctx, info_name="ls"):
            assert get_executed_command() == "flwr ls"


def test_get_executed_command_nested() -> None:
    """Test get_executed_command with nested commands (e.g., flwr federation list)."""
    # Create parent group "flwr" with child group "federation" and command "list"
    root_group = click.Group("flwr")
    federation_group = click.Group("federation")
    list_cmd = click.Command("list")

    with click.Context(root_group, info_name="flwr") as root_ctx:
        with click.Context(
            federation_group, parent=root_ctx, info_name="federation"
        ) as fed_ctx:
            with click.Context(list_cmd, parent=fed_ctx, info_name="list"):
                assert get_executed_command() == "flwr federation list"


@pytest.mark.parametrize("value", [None, 0, 1, 4, 2.5])
def test_optional_min_callback_accepts_none_and_valid_values(
    value: int | float | None,
) -> None:
    """Optional minimum validation should allow omitted and in-range values."""
    assert optional_min_callback(0)(value) == value


@pytest.mark.parametrize(("minimum", "value"), [(1, 0), (1, -1), (0, -0.5)])
def test_optional_min_callback_rejects_values_below_minimum(
    minimum: int, value: int | float
) -> None:
    """Optional minimum validation should reject explicit below-range values."""
    with pytest.raises(click.BadParameter, match=f"{minimum}"):
        optional_min_callback(minimum)(value)


def test_init_channel_from_connection_uses_resolved_connection() -> None:
    """Ensure resolved connection values are used for channel creation."""
    unresolved = SuperLinkConnection(
        name="local",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )
    resolved = SuperLinkConnection(
        name="local",
        address="127.0.0.1:9093",
        insecure=True,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )
    auth_plugin = Mock()
    auth_plugin.load_tokens = Mock()

    with patch(
        "flwr.cli.utils.ensure_local_superlink", return_value=resolved
    ) as mock_ensure:
        with patch("flwr.cli.utils.load_certificate_in_connection", return_value=None):
            with patch("flwr.cli.utils.create_channel") as mock_create:
                channel = Mock()
                mock_create.return_value = channel

                ret = init_channel_from_connection(unresolved, auth_plugin)

    assert ret is channel
    mock_ensure.assert_called_once_with(unresolved)
    auth_plugin.load_tokens.assert_called_once()

    kwargs = mock_create.call_args.kwargs
    assert kwargs["server_address"] == "127.0.0.1:9093"
    assert kwargs["insecure"] is True
    assert kwargs["root_certificates"] is None
    assert kwargs["max_message_length"] == GRPC_MAX_MESSAGE_LENGTH
    assert len(kwargs["interceptors"]) == 1
    channel.subscribe.assert_called_once()


def test_custom_grpc_err_handler() -> None:
    """Test flwr_cli_grpc_exc_handler with a custom error handler."""

    # Prepare
    class CustomError(Exception):
        """Custom error for testing."""

    mock_handler = Mock(side_effect=CustomError)
    grpc_error = grpc.RpcError()

    # Execute & assert
    with pytest.raises(CustomError):
        with flwr_cli_grpc_exc_handler(mock_handler):
            raise grpc_error

    mock_handler.assert_called_once_with(grpc_error)


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        (Path("a.py"), 0),
        (Path("d1/file.txt"), 1),
        (Path("d1/d2/d3/f.txt"), 3),
        (Path("d1/d2/d3/d4/d5/x"), 5),
    ],
)
def test_depth_of(rel: Path, expected: int) -> None:
    """Test the directory depth detection."""
    assert depth_of(rel) == expected


# === collect_project_files tests ===


def _make_files(root: Path, names: list[str]) -> None:
    """Create empty files under root, creating parent dirs as needed."""
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


def _rel(root: Path, paths: list[Path]) -> list[str]:
    """Return relative POSIX paths from a list of absolute paths."""
    return [p.relative_to(root).as_posix() for p in paths]


def test_collect_project_files_basic_include(tmp_path: Path) -> None:
    """Only files matching include patterns are returned."""
    _make_files(tmp_path, ["app.py", "utils.py", "readme.md", "data.csv"])
    paths = collect_project_files(
        tmp_path, include_patterns=("*.py",), exclude_patterns=()
    )
    assert _rel(tmp_path, paths) == ["app.py", "utils.py"]


def test_collect_project_files_exclude_patterns(tmp_path: Path) -> None:
    """Files matching exclude patterns are filtered out."""
    _make_files(tmp_path, ["app.py", "test.py", "secret.key"])
    paths = collect_project_files(
        tmp_path, include_patterns=("*",), exclude_patterns=("*.key",)
    )
    assert _rel(tmp_path, paths) == ["app.py", "test.py"]


def test_collect_project_files_gitignore_respected(tmp_path: Path) -> None:
    """Patterns in .gitignore are merged with exclude patterns."""
    _make_files(tmp_path, ["app.py", "debug.log", "build/output.bin"])
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
    paths = collect_project_files(
        tmp_path, include_patterns=("*",), exclude_patterns=()
    )
    assert _rel(tmp_path, paths) == [".gitignore", "app.py"]


def test_collect_project_files_max_depth_ok(tmp_path: Path) -> None:
    """Files within max_depth are accepted."""
    _make_files(tmp_path, ["root.py", "a/nested.py"])
    paths = collect_project_files(
        tmp_path, include_patterns=("*.py",), exclude_patterns=(), max_depth=1
    )
    assert _rel(tmp_path, paths) == ["a/nested.py", "root.py"]


def test_collect_project_files_max_depth_exceeded(tmp_path: Path) -> None:
    """ValueError is raised when a file exceeds max_depth."""
    _make_files(tmp_path, ["a/b/deep.py"])
    with pytest.raises(ValueError, match="exceeds the maximum directory depth"):
        collect_project_files(
            tmp_path, include_patterns=("*.py",), exclude_patterns=(), max_depth=1
        )


def test_collect_project_files_on_skip_called(tmp_path: Path) -> None:
    """on_skip callback is invoked for excluded files."""
    _make_files(tmp_path, ["keep.py", "skip.txt"])
    skipped: list[Path] = []
    collect_project_files(
        tmp_path,
        include_patterns=("*.py",),
        exclude_patterns=(),
        on_skip=skipped.append,
    )
    assert _rel(tmp_path, skipped) == ["skip.txt"]


def test_collect_project_files_sorted_deterministic(tmp_path: Path) -> None:
    """Results are sorted by POSIX relative path."""
    _make_files(tmp_path, ["z.py", "a.py", "m/b.py"])
    paths = collect_project_files(
        tmp_path, include_patterns=("*.py",), exclude_patterns=()
    )
    rel = _rel(tmp_path, paths)
    assert rel == sorted(rel)


def test_collect_project_files_empty_directory(tmp_path: Path) -> None:
    """An empty directory returns no files."""
    paths = collect_project_files(
        tmp_path, include_patterns=("*",), exclude_patterns=()
    )
    assert not paths


def test_collect_project_files_subdirectory_patterns(tmp_path: Path) -> None:
    """Include/exclude with subdirectory glob patterns."""
    _make_files(tmp_path, ["src/app.py", "src/test.py", "docs/guide.md"])
    paths = collect_project_files(
        tmp_path, include_patterns=("src/**",), exclude_patterns=("src/test.py",)
    )
    assert _rel(tmp_path, paths) == ["src/app.py"]


def test_collect_project_files_symlink_outside_root_excluded(tmp_path: Path) -> None:
    """Symlinks pointing outside root are excluded and trigger on_skip."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("sensitive data", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir()
    _make_files(project, ["app.py"])
    (project / "link.txt").symlink_to(outside / "secret.txt")

    skipped: list[Path] = []
    paths = collect_project_files(
        project, include_patterns=("*",), exclude_patterns=(), on_skip=skipped.append
    )
    assert _rel(project, paths) == ["app.py"]
    assert _rel(project, skipped) == ["link.txt"]
