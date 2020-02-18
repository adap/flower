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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module contains functions related to proto compilation"""

import glob
from os import path

import grpc_tools
from grpc_tools import protoc

DIR_PATH = path.dirname(path.realpath(__file__))
GRPC_PATH = grpc_tools.__path__[0]
IN_PATH = path.normpath(f"{DIR_PATH}/../../proto")
OUT_PATH = path.normpath(f"{DIR_PATH}/../flower/proto")
PROTO_FILES = glob.glob(f"{IN_PATH}/*.proto")


def compile_all():
    """Compile all protos in the proto directory into the src/flower/proto directory"""
    command = [
        "grpc_tools.protoc",
        f"--proto_path={GRPC_PATH}/_proto",  # path to google .proto fiels
        f"--proto_path={IN_PATH}",
        f"--python_out={OUT_PATH}",
        f"--grpc_python_out={OUT_PATH}",
        f"--mypy_out={OUT_PATH}",
    ] + PROTO_FILES

    exit_code = protoc.main(command)

    if exit_code != 0:
        raise Exception(f"Error: {command} failed")


if __name__ == "__main__":
    compile_all()
