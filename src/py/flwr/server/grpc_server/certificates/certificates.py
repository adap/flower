# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WIT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility for loading SSL credentials for SSL/TLS enabled gRPC server
tests."""
import subprocess
from os.path import abspath, dirname, join
from typing import Tuple

module_dir = dirname(abspath(__file__))


def get_absolute_path(filepath: str) -> str:
    """Return absolute path to file in directory of this module."""
    real_path = join(module_dir, filepath)
    return real_path


def get_paths() -> Tuple[str, str, str]:
    """Return path to all certificates required by gRPC server."""
    subprocess.run(["bash", "generate.sh"], check=True, cwd=module_dir)

    root_certificate = get_absolute_path("root.pem")
    certificate = get_absolute_path("localhost.crt")
    key = get_absolute_path("localhost.key")

    return root_certificate, certificate, key
