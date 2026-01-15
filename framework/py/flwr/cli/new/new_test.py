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
"""Test for Flower command line interface `new` command."""


import pytest
import typer

from .new import download_remote_app_via_api


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
def test_download_remote_app_via_api_rejects_invalid_formats(value: str) -> None:
    """For an invalid string, the function should fail fast with typer.Exit(code=1)."""
    with pytest.raises(typer.Exit) as exc:
        download_remote_app_via_api(value)

    # Ensure we specifically exited with code 1
    assert exc.value.exit_code == 1
