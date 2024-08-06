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
from logging import DEBUG, WARN
from typing import Dict, Optional

from flwr.common.logger import log
from flwr.common.typing import ConfigsRecordValues


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
    num_cpus : int (default: 2)
        Indicates the number of CPUs that a `ClientApp` needs when running.
    num_gpus : float (default: 0.0)
        Indicates the number of GPUs that a `ClientApp` needs when running. This
        value would normally be set based on the amount of VRAM a single `ClientApp`
        needs. It can be a decimal point value. For example, if `num_gpus=0.25` at
        most 4 `ClientApp` object could be run in parallel per GPU available and
        assuming 4x`num_cpus` are available in your system.
    """

    num_cpus: int = 2
    num_gpus: float = 0.0

    def __post_init__(self) -> None:
        """Validate resources after initialization."""
        self._validate()

    def _validate(self) -> None:

        if isinstance(self.num_cpus, float):
            num_cpus_int = int(self.num_cpus)
            log(
                WARN,
                "`num_cpus` for `ClientAppResources` needs to be an integer but a "
                "`float` was passed. It will be casted to `int`: (%s -> %s).",
                self.num_cpus,
                num_cpus_int,
            )
            self.num_cpus = num_cpus_int

        if self.num_cpus < 1:
            raise ValueError(
                "`num_cpus` for `ClientAppResources` needs to be an integer higher "
                f"than zero. You attempted to construct: {self}."
            )

        if not (isinstance(self.num_gpus, (int, float))) or self.num_gpus < 0.0:
            raise ValueError(
                "`num_gpus` for `ClientAppResources` needs to be an float higher "
                f"or equal than zero. You attempted to construct: {self}."
            )


@dataclass
class BackendConfig:
    """A config for a Simulation Engine backend.

    Parameters
    ----------
    name : str (default: ray)
        The name of the simulation Backend to use.
    clientapp_resources : Optional[ClientAppResources]
        A dataclass that indicates the sytem resources to should be assigned
        to a `ClientApp`. Higher resources per `ClientApp` means fewer can run
        in parallel.
    config: Optional[Dict[str, ConfigsRecordValues]]
        A dictionary used in the constructor of a backend.
    """

    name: str
    clientapp_resources: ClientAppResources
    config: Dict[str, ConfigsRecordValues]

    def __init__(
        self,
        name: str = "ray",
        clientapp_resources: Optional[ClientAppResources] = None,
        config: Optional[Dict[str, ConfigsRecordValues]] = None,
    ):
        self.name = name
        if clientapp_resources is None:
            # If unset, set default resources
            clientapp_resources = ClientAppResources()
            log(
                DEBUG,
                "The `BackendConfig` didn't receive `ClientAppResources`. "
                "The default resources will be used: %s",
                clientapp_resources,
            )

        self.clientapp_resources = clientapp_resources

        self.config = {} if config is None else config
