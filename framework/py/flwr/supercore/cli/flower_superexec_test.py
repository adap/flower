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
"""Tests for Flower SuperExec CLI argument parsing."""


import pytest

from flwr.supercore.version import package_version

from .flower_superexec import _parse_args


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_parse_superexec_version_flag(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The version flags should print the package version and exit."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_args().parse_args([flag])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"Flower version: {package_version}\n"
