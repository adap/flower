# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""This module contains functions related to proto compilation."""


import glob
from os import path

import grpc_tools
from grpc_tools import protoc

GRPC_PATH = grpc_tools.__path__[0]

DIR_PATH = path.dirname(path.realpath(__file__))
IN_PATH = path.normpath(f"{DIR_PATH}/../../proto")
OUT_PATH = path.normpath(f"{DIR_PATH}/..")
PROTO_FILES = glob.glob(f"{IN_PATH}/flwr/**/*.proto")


def compile_all() -> None:
    """Compile all protos in the `src/proto` directory.

    The directory structure of the `src/proto` directory will be mirrored in `src/py`.
    This is needed as otherwise `grpc_tools.protoc` will have broken imports.
    """
    command = [
        "grpc_tools.protoc",
        # Path to google .proto files
        f"--proto_path={GRPC_PATH}/_proto",
        # Path to root of our proto files
        f"--proto_path={IN_PATH}",
        # Output path
        f"--python_out={OUT_PATH}",
        f"--grpc_python_out={OUT_PATH}",
        f"--mypy_out={OUT_PATH}",
        f"--mypy_grpc_out={OUT_PATH}",
    ] + PROTO_FILES

    exit_code = protoc.main(command)

    if exit_code != 0:
        raise Exception(f"Error: {command} failed")  # pylint: disable=W0719


if __name__ == "__main__":
    compile_all()
