# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Long-running Flower client."""


import argparse
from logging import INFO

from flwr.common.logger import log


def run_client() -> None:
    """Run Flower client."""
    log(INFO, "Long-running Flower client starting")

    args = _parse_args_client().parse_args()

    print(args.server)


def _parse_args_client() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a long-running Flower client",
    )

    parser.add_argument(
        "--server",
        help="Server address",
        default="0.0.0.0:9092",
    )

    return parser
