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

import warnings
from dataclasses import dataclass
from logging import DEBUG, ERROR, WARN
from typing import Dict, Optional

from flwr.common.constant import (
    CLIEANT_APP_RESOURCES_DEFAULT_CPUS,
    CLIEANT_APP_RESOURCES_DEFAULT_GPUS,
)
from flwr.common.logger import log
from flwr.common.typing import ConfigsRecordValues

from . import Backend, error_messages_backends, supported_backends


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

    num_cpus: int = CLIEANT_APP_RESOURCES_DEFAULT_CPUS
    num_gpus: float = CLIEANT_APP_RESOURCES_DEFAULT_GPUS

    def __post_init__(self) -> None:
        """Validate resources after initialization."""
        self._validate()

    def _validate(self) -> None:

        if self.num_cpus is None:
            raise ValueError(
                "`num_cpus` must be a positive integer, but you passed `None`."
            )

        if self.num_gpus is None:
            raise ValueError("`num_cpus` must be a fractional number >=0.")

        if isinstance(self.num_cpus, float):
            num_cpus_int = int(self.num_cpus)
            warnings.warn(
                "`num_cpus` for `ClientAppResources` needs to be an integer but a "
                f"`float` was passed. It will be casted to `int`: ({self.num_cpus} -> "
                f"{num_cpus_int}).",
                UserWarning,
                stacklevel=1,
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


def get_backend_type_if_supported(backend_name: str) -> type[Backend]:
    """Check if backend is supported."""
    # Check if name of backend is in list of available backends
    log(DEBUG, "Supported backends: %s", list(supported_backends.keys()))
    if backend_name not in supported_backends:
        log(
            ERROR,
            "Backend `%s`, is not supported. Use any of %s or add support "
            "for a new backend.",
            backend_name,
            list(supported_backends.keys()),
        )
        if backend_name in error_messages_backends:
            log(ERROR, error_messages_backends[backend_name])
        raise ValueError(
            f"Unsupported backed `{backend_name}` when attempting to construct"
            "`BackendConfig`."
        )
    return supported_backends[backend_name]


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

    Examples
    --------
    In most situations, defining a `BackendConfig` primarily involves adjusting the
    amount of CPU/GPU resources that each of your `ClientApp` objects need to run
    efficiently. This can be done by setting the `ClientAppResources` structure taking
    into account both the compute/memory footprint of your `ClientApp` when doing work
    (e.g. when training a model on the local data the `ClientApp` owns) and the system
    resources available where your simulation runs.

    Adjusting `ClientAppResources` well often requires a few iterations. It's best to
    start with generous resources, run the simulation for a couple of rounds, and
    then adjust accordingly. If your system is underutilised that's usually a sign
    that you can lower the `ClientAppResources` specification and pack more `ClientApp`
    objects to run in parallel.

    >>> from flwr.simulation import BackendConfig, ClientAppResources
    >>>
    >>> # a lightweight ClientApp might run fine with just 1xCPU
    >>> # this will likely result in many `ClientApp`s running in parallel
    >>> client_resources = ClientAppResources(num_cpus=1, num_gpus=0)
    >>> backend_cfg = BackendConfig(clientapp_resources = client_resources)
    >>>
    >>> # More demanding workloads (e.g. training a larger model) in a `ClientApp` might
    >>> # need more resources. This will reduce the degree of parallelism so higher
    >>> # proportion of your system resources are accessible to a `ClientApp`. The
    >>> # example below shows how to set resources so at most two `ClientApp` run per
    >>> # GPU available.
    >>> client_resources = ClientAppResources(num_cpus=8, num_gpus=0.5)
    >>> backend_cfg = BackendConfig(clientapp_resources = client_resources)

    Some backends can be customized when they are initialized, i.e., when you start the
    simulation. This can be done via the `config` argument in a `BackendConfig` object.
    For example, for the `RayBackend`, you could disable the logging coming from the
    processes running the `ClientApp` objects like :

    >>> config = {"log_to_driver": False}
    >>> backend_cfg = BackendConfig(config=config)
    """

    name: str
    backend: type[Backend]
    clientapp_resources: ClientAppResources
    config: Dict[str, ConfigsRecordValues]

    def __init__(
        self,
        name: str = "ray",
        clientapp_resources: Optional[ClientAppResources] = None,
        config: Optional[Dict[str, ConfigsRecordValues]] = None,
    ):
        # Check backend is supported
        self.backend = get_backend_type_if_supported(name)
        self.name = name

        if clientapp_resources is None:
            # If unset, set default resources
            clientapp_resources = ClientAppResources()
            log(
                WARN,
                "`ClientAppResources` were not specified when constructing the "
                "`BackendConfig`. The default resources will be used: %s",
                clientapp_resources,
            )

        self.clientapp_resources = clientapp_resources

        self.config = {} if config is None else config
