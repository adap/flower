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
"""Tests for the CLI."""

from typer.testing import CliRunner

from flwr.supercore.version import package_version

from .app import app

runner = CliRunner()


def test_version_args() -> None:
    """Test the --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"Flower version: {package_version}\n" in result.output

    # Test the -V flag
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert f"Flower version: {package_version}\n" in result.output


def test_help_command() -> None:
    """Test the -h flag."""
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Options " in result.output
    assert "Commands " in result.output


def test_new_command() -> None:
    """Add appropriate assertions for the new command."""
    result = runner.invoke(app, ["new", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "new" in result.output


def test_run_command() -> None:
    """Add appropriate assertions for the run command."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "run" in result.output


def test_build_command() -> None:
    """Add appropriate assertions for the build command."""
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "build" in result.output


def test_install_command() -> None:
    """Add appropriate assertions for the install command."""
    result = runner.invoke(app, ["install", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "install" in result.output


def test_log_command() -> None:
    """Add appropriate assertions for the log command."""
    result = runner.invoke(app, ["log", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "log" in result.output


def test_ls_command() -> None:
    """Add appropriate assertions for the ls command."""
    result = runner.invoke(app, ["ls", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "ls" in result.output


def test_stop_command() -> None:
    """Add appropriate assertions for the stop command."""
    result = runner.invoke(app, ["stop", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "stop" in result.output


def test_login_command() -> None:
    """Add appropriate assertions for the login command."""
    result = runner.invoke(app, ["login", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "login" in result.output


def test_invalid_command() -> None:
    """Test CLI behavior with invalid commands and arguments."""
    # Test unknown command
    result = runner.invoke(app, ["nonexistent-command"])
    assert result.exit_code != 0
    assert "No such command" in result.output
