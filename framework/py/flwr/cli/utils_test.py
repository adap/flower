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
from typing import Any
from unittest.mock import patch

import typer

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    FLWR_DIR,
    FLWR_HOME,
    REFRESH_TOKEN_KEY,
)
from flwr.supercore.constant import (
    DEFAULT_FLOWER_CONFIG_TOML,
    FLOWER_CONFIG_FILE,
    SuperLinkConnectionTomlKey,
)
from flwr.supercore.typing import (
    SimulationBackendConfig,
    SimulationClientResources,
    SuperLinkConnection,
    SuperLinkSimulationOptions,
)

from .utils import (
    build_pathspec,
    get_sha256_hash,
    init_flwr_config,
    load_gitignore_patterns,
    parse_superlink_connection,
    read_superlink_connection,
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


class TestInitFlwrConfig(unittest.TestCase):
    """Test `init_flwr_config` function."""

    def test_init_flwr_config_creates_file(self) -> None:
        """Test that init_flwr_config creates the config file if it doesn't exist."""
        # Prepare
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Set FLWR_HOME to the temporary directory
            with patch.dict(os.environ, {FLWR_HOME: tmp_dir}):
                # Execute
                init_flwr_config()

                # Assert
                config_path = Path(tmp_dir) / "config.toml"
                self.assertTrue(config_path.exists())

                self.assertEqual(
                    config_path.read_text(encoding="utf-8"), DEFAULT_FLOWER_CONFIG_TOML
                )

    def test_init_flwr_config_does_not_overwrite(self) -> None:
        """Test that init_flwr_config does not overwrite existing config file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup - create existing config
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text("existing_content", encoding="utf-8")

            # Mock FLWR_HOME
            with patch.dict(os.environ, {FLWR_HOME: tmp_dir}):
                # Execute
                init_flwr_config()

                # Assert
                self.assertEqual(
                    config_path.read_text(encoding="utf-8"), "existing_content"
                )


class TestSuperLinkConnection(unittest.TestCase):
    """Unit tests for SuperLink connections."""

    def test_parse_superlink_connection_valid(self) -> None:
        """Test parse_superlink_connection with valid input."""
        # Prepare
        conn_dict = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:8080",
            SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: "root_cert.crt",
            SuperLinkConnectionTomlKey.INSECURE: False,
            SuperLinkConnectionTomlKey.ENABLE_ACCOUNT_AUTH: True,
        }
        name = "test_service"

        # Execute
        config = parse_superlink_connection(conn_dict, name)

        # Assert
        self.assertEqual(config.name, name)
        self.assertEqual(config.address, "127.0.0.1:8080")
        self.assertEqual(config.root_certificates, "root_cert.crt")
        self.assertFalse(config.insecure)
        self.assertTrue(config.enable_account_auth)

    def test_parse_superlink_connection_invalid(self) -> None:
        """Test parse_superlink_connection with invalid input."""
        # Missing required fields
        conn_dict = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:8080",
        }
        name = "test_service"

        with self.assertRaises(ValueError):
            parse_superlink_connection(conn_dict, name)

    def test_parse_superlink_connection_invalid_type(self) -> None:
        """Test parse_superlink_connection with invalid type."""
        conn_dict = {
            SuperLinkConnectionTomlKey.ADDRESS: 123,  # Invalid type, should be str
        }
        name = "test_service"

        with self.assertRaisesRegex(
            ValueError, "Invalid value for key 'address': expected str, but got int"
        ):
            parse_superlink_connection(conn_dict, name)

    def test_parse_superlink_connection_valid_cases(self) -> None:
        """Test various valid connection configurations from valid.toml."""
        # Case 1: SuperLink with address and enable-account-auth
        conn_dict_1 = {
            SuperLinkConnectionTomlKey.ADDRESS: "supergrid.flower.ai",
            SuperLinkConnectionTomlKey.ENABLE_ACCOUNT_AUTH: True,
        }
        config_1 = parse_superlink_connection(conn_dict_1, "supergrid")
        self.assertEqual(config_1.address, "supergrid.flower.ai")
        self.assertTrue(config_1.enable_account_auth)
        self.assertIsNone(config_1.options)

        # Case 2: Local PoC with address and insecure
        conn_dict_2 = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:9093",
            SuperLinkConnectionTomlKey.INSECURE: True,
        }
        config_2 = parse_superlink_connection(conn_dict_2, "local-poc")
        self.assertEqual(config_2.address, "127.0.0.1:9093")
        self.assertTrue(config_2.insecure)

        # Case 3: Local PoC Dev with address and root-certificates
        conn_dict_3 = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:9093",
            SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: "root_cert.crt",
        }
        config_3 = parse_superlink_connection(conn_dict_3, "local-poc-dev")
        self.assertEqual(config_3.address, "127.0.0.1:9093")
        self.assertEqual(config_3.root_certificates, "root_cert.crt")

        # Case 4: Remote Sim with address, root-certificates, and options
        conn_dict_4 = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:9093",
            SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: "root_cert.crt",
            SuperLinkConnectionTomlKey.OPTIONS: {
                "num-supernodes": 10,
                "backend": {
                    "client-resources": {"num-cpus": 1},
                },
            },
        }
        config_4 = parse_superlink_connection(conn_dict_4, "remote-sim")
        self.assertEqual(config_4.address, "127.0.0.1:9093")
        self.assertEqual(config_4.root_certificates, "root_cert.crt")
        self.assertIsNotNone(config_4.options)
        assert config_4.options is not None
        self.assertEqual(config_4.options.num_supernodes, 10)

    def test_parse_superlink_connection_simulation(self) -> None:
        """Test parse_superlink_connection with simulation options."""
        # Prepare
        conn_dict = {
            SuperLinkConnectionTomlKey.OPTIONS: {
                "num-supernodes": 10,
                "backend": {
                    "client-resources": {"num-cpus": 1.0, "num-gpus": 0.5},
                    "init-args": {"logging-level": "info", "log-to-drive": True},
                    "name": "custom-backend",
                },
            }
        }
        name = "local-simulation"

        # Execute
        config = parse_superlink_connection(conn_dict, name)

        # Assert
        self.assertEqual(config.name, name)
        self.assertIsNone(config.address)
        self.assertIsNotNone(config.options)
        assert config.options is not None
        self.assertEqual(config.options.num_supernodes, 10)

        # Check Backend
        backend = config.options.backend
        self.assertIsNotNone(backend)
        assert backend is not None
        self.assertIsNotNone(backend.client_resources)
        assert backend.client_resources is not None
        self.assertEqual(backend.client_resources.num_cpus, 1.0)
        self.assertEqual(backend.client_resources.num_gpus, 0.5)

        self.assertIsNotNone(backend.init_args)
        assert backend.init_args is not None
        self.assertEqual(backend.init_args.logging_level, "info")
        self.assertTrue(backend.init_args.log_to_drive)
        self.assertEqual(backend.name, "custom-backend")

    def test_parse_superlink_connection_simulation_invalid_name(self) -> None:
        """Test parse_superlink_connection with invalid name type."""
        # Prepare
        conn_dict = {
            "options": {
                "num-supernodes": 10,
                "backend": {
                    "name": 123,  # Invalid type, should be str
                },
            }
        }
        name = "local-simulation"

        # Execute & Assert
        with self.assertRaisesRegex(ValueError, "backend.name must be a string"):
            parse_superlink_connection(conn_dict, name)

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_defaults(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_superlink_connection uses default when no arg provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service-2"

            [superlink.mock-service]
            address = "losthost:1234"
            insecure = false
            enable-account-auth = false

            [superlink.mock-service-2]
            address = "losthost:9093"
            insecure = true
            enable-account-auth = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_superlink_connection()

            # Assert
            assert config is not None
            self.assertEqual(config.name, "mock-service-2")
            self.assertEqual(config.address, "losthost:9093")

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_explicit(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_superlink_connection with explicit connection name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service-2"

            [superlink.mock-service]
            address = "losthost:1234"
            insecure = false
            enable-account-auth = true

            [superlink.mock-service-2]
            address = "losthost:9093"
            insecure = true
            enable-account-auth = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_superlink_connection("mock-service")

            # Assert
            assert config is not None
            self.assertEqual(config.name, "mock-service")
            self.assertEqual(config.address, "losthost:1234")

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_explicit_missing(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_superlink_connection with explicit but missing connection."""
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
                read_superlink_connection("missing-service")

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_no_default_failure(
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
                read_superlink_connection()

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_default_missing_connection(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test failure when default is set but the connection block is missing."""
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
                read_superlink_connection()

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_no_file(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_superlink_connection when file does not exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)

            config = read_superlink_connection()

            self.assertIsNone(config)

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_corrupted(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_superlink_connection when file is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Write invalid TOML
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("invalid = toml [ [")

            with self.assertRaises(typer.Exit):
                read_superlink_connection()

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_simulation(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_superlink_connection with simulation profile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "local-sim"

            [superlink.local-sim]
            options.num-supernodes = 2
            options.backend.client-resources.num-cpus = 2.0
            options.backend.init-args.num-cpus = 1
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_superlink_connection()

            # Assert
            assert config is not None
            self.assertEqual(config.name, "local-sim")
            self.assertIsNone(config.address)
            self.assertIsNotNone(config.options)
            assert config.options is not None
            self.assertEqual(config.options.num_supernodes, 2)
            self.assertIsNotNone(config.options.backend)
            assert config.options.backend is not None
            self.assertIsNotNone(config.options.backend.client_resources)
            assert config.options.backend.client_resources is not None
            self.assertEqual(config.options.backend.client_resources.num_cpus, 2.0)
            self.assertIsNotNone(config.options.backend.init_args)
            assert config.options.backend.init_args is not None
            self.assertEqual(config.options.backend.init_args.num_cpus, 1)
            self.assertEqual(config.options.backend.name, "ray")

    @patch("flwr.cli.utils.get_flwr_home")
    def test_read_superlink_connection_simulation_unknown_key(
        self, mock_get_flwr_home: unittest.mock.Mock
    ) -> None:
        """Test read_superlink_connection with unknown keys (should be ignored)."""
        # Prepare
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_get_flwr_home.return_value = Path(tmp_dir)
            config_path = os.path.join(tmp_dir, "config.toml")
            toml_content = """
            [superlink]
            default = 'local-sim'

            [superlink.local-sim]
            options.num-supernodes = 2
            options.backend.client-resources.num-cpus = 2.0
            options.backend.unknown-key = "unexpected"
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            name = "local-sim"
            # Expected: a valid config that doesn't include keys
            # that don't align with the scheema
            expected_config = SuperLinkConnection(
                name="local-sim",
                options=SuperLinkSimulationOptions(
                    num_supernodes=2,
                    backend=SimulationBackendConfig(
                        client_resources=SimulationClientResources(num_cpus=2.0),
                    ),
                ),
            )

            # Execute
            config = read_superlink_connection(name)

            # Assert
            self.assertEqual(config, expected_config)

    def test_parse_superlink_connection_mixed(self) -> None:
        """Test with both address and options (should be valid)."""
        # Prepare
        conn_dict = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:8080",
            SuperLinkConnectionTomlKey.INSECURE: True,
            "options": {"num-supernodes": 5},
        }
        name = "mixed-connection"

        # Execute
        config = parse_superlink_connection(conn_dict, name)

        # Assert
        self.assertEqual(config.address, "127.0.0.1:8080")
        self.assertIsNotNone(config.options)
        assert config.options is not None
        self.assertEqual(config.options.num_supernodes, 5)

    def test_parse_superlink_connection_mixed_invalid(self) -> None:
        """Test mixed connection missing required simulation field."""
        # Prepare
        conn_dict: dict[str, Any] = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:8080",
            SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: None,
            SuperLinkConnectionTomlKey.INSECURE: False,
            SuperLinkConnectionTomlKey.ENABLE_ACCOUNT_AUTH: False,
            "options": {},  # Missing num-supernodes
        }
        name = "mixed-invalid"

        # Execute & Assert
        with self.assertRaisesRegex(
            ValueError, "Invalid simulation options: num-supernodes must be an integer"
        ):
            parse_superlink_connection(conn_dict, name)
