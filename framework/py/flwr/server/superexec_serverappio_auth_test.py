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
"""Tests for SuperExec auth argument validation in server app."""


import argparse
from unittest.mock import patch

import pytest

from flwr.common.constant import (
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
    ExecPluginType,
)
from flwr.common.exit import ExitCode
from flwr.server.app import _validate_superexec_auth_settings
from flwr.server.superlink.superexec_auth import (
    SuperExecAuthConfig,
    get_disabled_superexec_auth_config,
)


def _enabled_superexec_auth_config() -> SuperExecAuthConfig:
    return SuperExecAuthConfig(
        enabled=True,
        timestamp_tolerance_sec=300,
        allowed_public_keys={
            ExecPluginType.SERVER_APP: {b"serverapp-public-key"},
            ExecPluginType.SIMULATION: set(),
        },
    )


def test_superexec_private_key_requires_auth_config() -> None:
    """Reject passing private key path without SuperExec auth config path."""
    args = argparse.Namespace(
        auth_superexec_private_key="/tmp/superexec.key",
        superexec_auth_config=None,
        isolation=ISOLATION_MODE_PROCESS,
    )

    with patch(
        "flwr.server.app.flwr_exit", side_effect=RuntimeError("exit")
    ) as mock_exit:
        with pytest.raises(RuntimeError, match="exit"):
            _validate_superexec_auth_settings(
                args, get_disabled_superexec_auth_config()
            )

    mock_exit.assert_called_once()
    assert mock_exit.call_args.args[0] == ExitCode.SUPERLINK_INVALID_ARGS
    assert "`--superexec-auth-config`" in mock_exit.call_args.args[1]


def test_superexec_enabled_in_subprocess_requires_private_key() -> None:
    """Reject enabled SuperExec auth in subprocess mode without private key."""
    args = argparse.Namespace(
        auth_superexec_private_key=None,
        superexec_auth_config="/tmp/superexec-auth.yaml",
        isolation=ISOLATION_MODE_SUBPROCESS,
    )

    with patch(
        "flwr.server.app.flwr_exit", side_effect=RuntimeError("exit")
    ) as mock_exit:
        with pytest.raises(RuntimeError, match="exit"):
            _validate_superexec_auth_settings(args, _enabled_superexec_auth_config())

    mock_exit.assert_called_once()
    assert mock_exit.call_args.args[0] == ExitCode.SUPERLINK_INVALID_ARGS
    assert "no SuperExec private key path" in mock_exit.call_args.args[1]


def test_superexec_disabled_logs_warning() -> None:
    """Log disabled warning when SuperExec auth is disabled."""
    args = argparse.Namespace(
        auth_superexec_private_key=None,
        superexec_auth_config=None,
        isolation=ISOLATION_MODE_PROCESS,
    )

    with (
        patch("flwr.server.app.flwr_exit") as mock_exit,
        patch("flwr.server.app.log") as mock_log,
    ):
        _validate_superexec_auth_settings(args, get_disabled_superexec_auth_config())

    mock_exit.assert_not_called()
    mock_log.assert_called_once()
    assert "SuperExec authentication is disabled" in mock_log.call_args.args[1]


def test_superexec_enabled_logs_info() -> None:
    """Log enabled info when SuperExec auth is enabled."""
    args = argparse.Namespace(
        auth_superexec_private_key=None,
        superexec_auth_config="/tmp/superexec-auth.yaml",
        isolation=ISOLATION_MODE_PROCESS,
    )

    with (
        patch("flwr.server.app.flwr_exit") as mock_exit,
        patch("flwr.server.app.log") as mock_log,
    ):
        _validate_superexec_auth_settings(args, _enabled_superexec_auth_config())

    mock_exit.assert_not_called()
    mock_log.assert_called_once()
    assert "SuperExec authentication is enabled" in mock_log.call_args.args[1]


def test_superexec_private_key_warns_in_process_mode() -> None:
    """Warn that private key is ignored in process isolation mode."""
    args = argparse.Namespace(
        auth_superexec_private_key="/tmp/superexec.key",
        superexec_auth_config="/tmp/superexec-auth.yaml",
        isolation=ISOLATION_MODE_PROCESS,
    )

    with (
        patch("flwr.server.app.flwr_exit") as mock_exit,
        patch("flwr.server.app.log") as mock_log,
    ):
        _validate_superexec_auth_settings(args, _enabled_superexec_auth_config())

    mock_exit.assert_not_called()
    assert any(
        "ignored when `--isolation=process`" in call.args[1]
        for call in mock_log.call_args_list
    )
