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
import unittest.mock
from pathlib import Path

from flwr.cli.utils import (
    build_pathspec,
    get_sha256_hash,
    init_main_config,
    load_gitignore_patterns,
    validate_credentials_content,
)
from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    FLWR_DIR,
    FLWR_HOME,
    REFRESH_TOKEN_KEY,
)
from flwr.supercore.constant import DEFAULT_CONFIG_TOML


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


class TestInitMainConfig(unittest.TestCase):
    """Test `init_main_config` function."""

    def test_init_main_config_creates_file(self) -> None:
        """Test that init_main_config creates the config file if it doesn't exist."""
        # Prepare
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Set FLWR_HOME to the temporary directory
            with unittest.mock.patch.dict(os.environ, {FLWR_HOME: tmp_dir}):
                # Execute
                init_main_config()

                # Assert
                config_path = Path(tmp_dir) / "config.toml"
                self.assertTrue(config_path.exists())

                self.assertEqual(
                    config_path.read_text(encoding="utf-8"), DEFAULT_CONFIG_TOML
                )

    def test_init_main_config_does_not_overwrite(self) -> None:
        """Test that init_main_config does not overwrite existing config file."""
        # Prepare
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            existing_content = "existing_content"
            config_path.write_text(existing_content, encoding="utf-8")

            # Set FLWR_HOME to the temporary directory
            with unittest.mock.patch.dict(os.environ, {FLWR_HOME: tmp_dir}):
                init_main_config()

                # Assert
                self.assertEqual(
                    config_path.read_text(encoding="utf-8"), existing_content
                )
