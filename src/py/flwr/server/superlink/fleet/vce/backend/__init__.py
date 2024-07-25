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
"""Simulation Engine Backends."""

import importlib
from typing import Dict, Type

from .backend import Backend, BackendConfig

is_ray_installed = importlib.util.find_spec("ray") is not None

# Mapping of supported backends
supported_backends: Dict[str, Type[Backend]] = {}

# To log backend-specific error message when chosen backend isn't available
error_messages_backends: Dict[str, str] = {}

if is_ray_installed:
    from .raybackend import RayBackend

    supported_backends["ray"] = RayBackend
else:
    error_messages_backends[
        "ray"
    ] = """Unable to import module `ray`.

    To install the necessary dependencies, install `flwr` with the `simulation` extra:

        pip install -U "flwr[simulation]"
    """


__all__ = [
    "Backend",
    "BackendConfig",
]
