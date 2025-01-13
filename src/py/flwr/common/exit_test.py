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
"""Tests for the unified exit function."""


from .exit import ExitCode, EXIT_CODE_HELP
from pathlib import Path


def test_exit_code_help_exist() -> None:
    """Test if all exit codes have help message."""
    for name, value in ExitCode.__dict__.items():
        if name.startswith("__"):
            continue
        assert value in EXIT_CODE_HELP, f"Exit code {name} ({value}) does not have help message."


def test_exit_code_help_url_exist() -> None:
    """Test if all exit codes have help URL."""
    # Get all exit code help URLs
    dir_path = Path("framework/docs/source/exit-codes")
    help_urls = {int(f.stem) for f in dir_path.glob("*.rst") if f.stem[0] != "_"}
    
    # Check if all exit codes have help URL
    for name, value in ExitCode.__dict__.items():
        if name.startswith("__"):
            continue
        assert value in help_urls, f"Exit code {name} ({value}) does not have help URL."
