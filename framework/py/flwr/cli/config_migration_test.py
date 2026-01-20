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
"""Tests for config migration."""


import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tomli
from parameterized import parameterized

from .config_migration import (
    CONFIG_MIGRATION_NOTICE,
    _comment_out_legacy_toml_config,
    _is_legacy_usage,
    _migrate_pyproject_toml_to_flower_config,
)
from .flower_config import init_flwr_config, read_superlink_connection

TEST_PYPROJECT_TOML = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "demo"
version = "1.0.0"
dependencies = ["flwr[simulation]>=1.19.0"]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "pan"

[tool.flwr.app.components]
serverapp = "demo.server_app:app"
clientapp = "demo.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 30

[tool.flwr.federations]
default = "local-poc"

[tool.flwr.federations.local-poc]
address = "127.0.0.1:9093"
insecure = true

[tool.flwr.federations.my-sim]
options.num-supernodes = 2

[tool.flwr.federations.researchgrid]
address = "researchgrid.flower.blue"
root-certificates = "certs/researchgrid.crt"
enable-account-auth = true
options.num-supernodes = 2
"""


class TestConfigMigration(unittest.TestCase):
    """Tests for config migration helpers."""

    def setUp(self) -> None:
        """Set up temporary Flower home directory for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732

        # Patch get_flwr_home to point to a temporary Flower home directory
        self.home_path = Path(self.temp_dir.name)
        self.home_path.mkdir(parents=True, exist_ok=True)
        self.get_home_patcher = patch(
            "flwr.cli.flower_config.get_flwr_home", return_value=self.home_path
        )
        self.get_home_patcher.start()
        init_flwr_config()

        # Create a temporary app directory with pyproject.toml
        self.app_path = self.home_path / "test-app"
        self.app_path.mkdir(parents=True, exist_ok=True)
        (self.app_path / "pyproject.toml").write_text(TEST_PYPROJECT_TOML)

    def tearDown(self) -> None:
        """Tear down temporary Flower home directory."""
        self.get_home_patcher.stop()
        self.temp_dir.cleanup()

    @parameterized.expand(  # type: ignore[misc]
        [
            ("local", ["my-federation"], True),  # single extra arg
            ("/abs/path", [], True),  # absolute path
            ("./relative/path", [], True),  # relative path with slash
            (".", [], True),  # dot path
            ("named-conn", [], False),  # normal name
        ]
    )
    def test_is_legacy_usage(
        self, superlink: str, args: list[str], expected: bool
    ) -> None:
        """Test `_is_legacy_usage` function."""
        assert _is_legacy_usage(superlink, args) is expected

    def test_migrate_pyproject_toml_to_flower_config(self) -> None:
        """Test `_migrate_pyproject_toml_to_flower_config` function."""
        # Execute
        _migrate_pyproject_toml_to_flower_config(self.app_path, None)
        default_conn = read_superlink_connection()

        # Assert default is set from legacy config (local-poc)
        assert default_conn is not None
        assert default_conn.name == "local-poc"

        # Assert migrated connections are readable
        local_poc = read_superlink_connection("local-poc")
        assert local_poc is not None
        assert local_poc.address == "127.0.0.1:9093"
        assert local_poc.insecure is True

        my_sim = read_superlink_connection("my-sim")
        assert my_sim is not None
        assert my_sim.options is not None
        assert my_sim.options.num_supernodes == 2

        researchgrid = read_superlink_connection("researchgrid")
        assert researchgrid is not None
        assert researchgrid.address == "researchgrid.flower.blue"
        assert researchgrid.root_certificates == str(
            (self.app_path / "certs/researchgrid.crt").resolve()
        )
        assert researchgrid.enable_account_auth is True

    def test_comment_out_legacy_toml_config(self) -> None:
        """Test `_comment_out_legacy_toml_config` function."""
        # Execute
        _comment_out_legacy_toml_config(self.app_path)

        pyproject_content = (self.app_path / "pyproject.toml").read_text()

        # Notice is added
        assert CONFIG_MIGRATION_NOTICE in pyproject_content

        # After commenting, tool.flwr.federations should be absent when parsed
        parsed = tomli.loads(pyproject_content)
        assert "tool" in parsed
        assert "flwr" in parsed["tool"]
        assert "federations" not in parsed["tool"]["flwr"]

        # The app section should remain intact
        assert parsed["tool"]["flwr"]["app"]["publisher"] == "pan"
