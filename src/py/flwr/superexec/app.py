# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower SuperExec app."""


import argparse

from flwr.common.object_ref import load_app, validate

from .executor import Executor


def load_executor(
    args: argparse.Namespace,
) -> Executor:
    """Get the executor plugin."""
    executor_ref: str = args.executor
    valid, error_msg = validate(executor_ref, project_dir=args.executor_dir)
    if not valid and error_msg:
        raise LoadExecutorError(error_msg) from None

    executor = load_app(executor_ref, LoadExecutorError, args.executor_dir)

    if not isinstance(executor, Executor):
        raise LoadExecutorError(
            f"Attribute {executor_ref} is not of type {Executor}",
        ) from None

    return executor


class LoadExecutorError(Exception):
    """Error when trying to load `Executor`."""
