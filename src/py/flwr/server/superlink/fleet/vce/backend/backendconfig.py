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
"""Backend config."""


from dataclasses import dataclass
from typing import Optional

from flwr.common.typing import UserConfig


@dataclass
class ClientAppResources:
    """Resources for a `ClientApp`.

    These resources are used by a `Backend` to control the degree of parallelism
    of a simulation. Lower `num_cpus` and `num_gpus` allow for running more
    `ClientApp` objects in parallel. Note system resources are shared among all
    `ClientApp`s running. In other words, per-`ClientApp` resource limits are not
    enforced and are used exclusively as a guide to control parallelism.

    Parameters
    ----------
    num_cpus : float (default: 1.0)
        Indicates the number of CPUs that a `ClientApp` needs to be executed.
    num_gpus : float (default: 0.0)
        Indicates the number of GPUs that a `ClientApp` needs to be executed. This
        value would normally be set based on the amount of VRAM a single `ClientApp`
        needs. It can be a decimal point value. For example, if `num_gpus=0.25` at
        most 4 `ClientApp` object could be run in parallel per GPU available and
        asuming 4x`num_cpus` are available in your system.
    """

    num_cpus: float = 1.0
    num_gpus: float = 0.0


@dataclass
class BackendConfig:
    """A config for a Simulation Engine backend."""

    name: str
    clientapp_resources: ClientAppResources
    config: Optional[UserConfig]

    def __init__(
        self,
        name: str = "ray",
        clientapp_resources: Optional[ClientAppResources] = None,
        config: Optional[UserConfig] = None,
    ):
        self.name = name
        if clientapp_resources is None:
            # If unset, set default resources
            clientapp_resources = ClientAppResources()

        self.clientapp_resources = clientapp_resources

        self.config = {} if config is None else config
