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
import sys
from logging import INFO

from uvicorn.importer import import_from_string

from flwr.app import Flower
from flwr.client import start_client
from flwr.common.logger import log


def run_client() -> None:
    """Run Flower client."""
    log(INFO, "Long-running Flower client starting")

    args = _parse_args_client().parse_args()

    print(args.server)
    print(args.app_dir)
    print(args.app)

    app_dir = args.app_dir
    if app_dir is not None:
        sys.path.insert(0, app_dir)

    def _load() -> Flower:
        app: Flower = import_from_string(args.app)
        return app

    return start_client(
        server_address=args.server,
        load_app_fn=_load,
        transport="grpc-rere",  # Only
    )


def _parse_args_client() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a long-running Flower client",
    )

    parser.add_argument(
        "--server",
        default="0.0.0.0:9092",
        help="Server address",
    )

    parser.add_argument(
        "--app-dir",
        default="",
        help="Look for APP in specified directory, by adding this to the PYTHONPATH."
        " Defaults to the current working directory.",
    )

    parser.add_argument(
        "--app",
        help="",
    )

    return parser
