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
"""Tests for SuperExec base plugin command construction."""


from unittest.mock import Mock, patch

from flwr.supercore.superexec.plugin.base_exec_plugin import BaseExecPlugin


class DummyExecPlugin(BaseExecPlugin):
    """Minimal plugin for testing command construction."""

    command = "dummy-app"
    appio_api_address_arg = "--appio-api-address"


def test_launch_app_forwards_runtime_dependency_install_and_index_url() -> None:
    """Ensure app launch forwards runtime install flags."""
    plugin = DummyExecPlugin(
        appio_api_address="127.0.0.1:9091",
        get_run=Mock(),
        index_url="http://127.0.0.1:3141/root/pypi/+simple/",
        runtime_dependency_install=True,
    )

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_exec_plugin.os.getpid",
            return_value=1234,
        ),
        patch(
            "flwr.supercore.superexec.plugin.base_exec_plugin.subprocess.Popen"
        ) as popen,
    ):
        plugin.launch_app(token="token-123", run_id=7)

    assert popen.call_args.args[0] == [
        "dummy-app",
        "--insecure",
        "--appio-api-address",
        "127.0.0.1:9091",
        "--token",
        "token-123",
        "--parent-pid",
        "1234",
        "--allow-runtime-dependency-installation",
        "--index-url",
        "http://127.0.0.1:3141/root/pypi/+simple/",
    ]


def test_launch_app_skips_optional_runtime_flags_by_default() -> None:
    """Ensure app launch omits optional runtime install flags by default."""
    plugin = DummyExecPlugin(
        appio_api_address="127.0.0.1:9091",
        get_run=Mock(),
    )

    with patch(
        "flwr.supercore.superexec.plugin.base_exec_plugin.subprocess.Popen"
    ) as popen:
        plugin.launch_app(token="token-123", run_id=7)

    assert "--allow-runtime-dependency-installation" not in popen.call_args.args[0]
    assert "--index-url" not in popen.call_args.args[0]
