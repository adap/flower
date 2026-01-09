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
from unittest.mock import patch

import pytest
import typer

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    FLWR_DIR,
    REFRESH_TOKEN_KEY,
)
from flwr.supercore.constant import FLOWER_CONFIG_FILE, SuperlinkProfileTomlKey

from .utils import (
    build_pathspec,
    get_sha256_hash,
    load_gitignore_patterns,
    parse_app_spec,
    parse_superlink_profile,
    read_current_superlink_profile,
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


@pytest.mark.parametrize(
    "value",
    [
        "user/app==1.2.3",  # missing '@'
        "@accountapp==1.2.3",  # missing slash
        "@account/app==1.2",  # bad version
        "@account/app==1.2.3.4",  # bad version
        "@account*/app==1.2.3",  # bad user id chars
        "@account/app*==1.2.3",  # bad app id chars
    ],
)
def test_parse_app_spec_rejects_invalid_formats(value: str) -> None:
    """For an invalid string, the function should fail fast with typer.Exit(code=1)."""
    with pytest.raises(typer.Exit) as exc:
        parse_app_spec(value)

    # Ensure we specifically exited with code 1
    assert exc.value.exit_code == 1


class TestSuperLinkProfile(unittest.TestCase):
    """Unit tests for SuperLink profiles."""

    def test_parse_superlink_profile_valid(self) -> None:
        """Test parse_superlink_profile with valid input."""
        # Prepare
        conn_dict = {
            SuperlinkProfileTomlKey.ADDRESS: "127.0.0.1:8080",
            SuperlinkProfileTomlKey.ROOT_CERTIFICATES: "root_cert.crt",
            SuperlinkProfileTomlKey.INSECURE: False,
            SuperlinkProfileTomlKey.ENABLE_ACCOUNT_AUTH: True,
        }
        service = "test_service"

        # Execute
        config = parse_superlink_profile(conn_dict, service)

        # Assert
        self.assertEqual(config.service, service)
        self.assertEqual(config.address, "127.0.0.1:8080")
        self.assertEqual(config.root_certificates, "root_cert.crt")
        self.assertFalse(config.insecure)
        self.assertTrue(config.enable_account_auth)

    def test_parse_superlink_profile_invalid(self) -> None:
        """Test parse_superlink_profile with invalid input."""
        # Missing required fields
        conn_dict = {
            SuperlinkProfileTomlKey.ADDRESS: "127.0.0.1:8080",
        }
        service = "test_service"

        with self.assertRaises(ValueError):
            parse_superlink_profile(conn_dict, service)

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_current_superlink_profile_defaults(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_current_superlink_profile uses default when no arg provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service-2"

            [superlink.mock-service]
            address = "losthost:9093"
            insecure = false
            enable-account-auth = false

            [superlink.mock-service-2]
            address = "losthost:9094"
            insecure = true
            enable-account-auth = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_current_superlink_profile()

            # Assert
            assert config is not None
            self.assertEqual(config.service, "mock-service-2")
            self.assertEqual(config.address, "losthost:9094")

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_current_superlink_profile_explicit(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_current_superlink_profile with explicit profile name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service-2"

            [superlink.mock-service]
            address = "losthost:9093"
            insecure = false
            enable-account-auth = false

            [superlink.mock-service-2]
            address = "losthost:9094"
            insecure = true
            enable-account-auth = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_current_superlink_profile("mock-service")

            # Assert
            assert config is not None
            self.assertEqual(config.service, "mock-service")
            self.assertEqual(config.address, "losthost:9093")

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_current_superlink_profile_explicit_missing(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_current_superlink_profile with explicit but missing profile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service"

            [superlink.mock-service]
            address = "losthost:9093"
            insecure = false
            enable-account-auth = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute & Assert
            with self.assertRaises(typer.Exit):
                read_current_superlink_profile("missing-service")

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_current_superlink_profile_no_default_failure(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test failure when no default is set and no arg provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content without default
            toml_content = """
            [superlink]
            # No default

            [superlink.mock-service]
            address = "losthost:9093"
            insecure = false
            enable-account-auth = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute & Assert
            with self.assertRaises(typer.Exit):
                read_current_superlink_profile()

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_current_superlink_profile_default_missing_profile(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test failure when default is set but the profile block is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Default points to "missing-service", which is not defined
            toml_content = """
            [superlink]
            default = "missing-service"

            [superlink.other-service]
            address = "losthost:9093"
            insecure = false
            enable-account-auth = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute & Assert
            with self.assertRaises(typer.Exit):
                read_current_superlink_profile()

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_current_superlink_profile_no_file(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_current_superlink_profile when file does not exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)

            config = read_current_superlink_profile()

            self.assertIsNone(config)

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_current_superlink_profile_corrupted(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_current_superlink_profile when file is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Write invalid TOML
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("invalid = toml [ [")

            with self.assertRaises(typer.Exit):
                read_current_superlink_profile()
