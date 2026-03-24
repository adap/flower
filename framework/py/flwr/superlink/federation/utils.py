# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Utilities for federation management."""


from flwr.proto.federation_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.supercore.constant import DEFAULT_SIMULATION_CONFIG


def get_default_simulation_config() -> SimulationConfig:
    """Return the default simulation configuration."""
    config = SimulationConfig()
    config.CopyFrom(DEFAULT_SIMULATION_CONFIG)
    return config
