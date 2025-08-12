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
"""Simple Flower SuperExec plugin for simulation processes."""


from .base_exec_plugin import BaseExecPlugin


class SimulationExecPlugin(BaseExecPlugin):
    """Simple Flower SuperExec plugin for simulation processes.

    The plugin always selects the first candidate run ID.
    """

    command = "flwr-simulation"
    appio_api_address_arg = "--simulationio-api-address"
