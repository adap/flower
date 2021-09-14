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


def load() -> Tuple[str, str, str]:
    """Start SSL/TLS enabled server."""
    # Trigger script which generates the certificates
    subprocess.run(["bash", "generate.sh"], check=True, cwd=module_dir)

    ssl_files = (
        join(module_dir, "ca.cert"),
        join(module_dir, "server.pem"),
        join(module_dir, "server.key"),
    )

    return ssl_files
