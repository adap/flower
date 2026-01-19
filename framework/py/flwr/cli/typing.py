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
"""Flower command line interface type definitions."""


from dataclasses import dataclass
from pathlib import Path

from flwr.cli.constant import (
    DEFAULT_SIMULATION_BACKEND_NAME,
    SuperLinkConnectionTomlKey,
)


@dataclass
class SimulationClientResources:
    """Resource configuration for the ClientApp."""

    num_cpus: float | None = None
    num_gpus: float | None = None

    def __post_init__(self) -> None:
        """Validate client resources configuration."""
        if self.num_cpus is not None and not isinstance(self.num_cpus, (int, float)):
            raise ValueError("client-resources.num-cpus must be a number (int/float).")
        if self.num_gpus is not None and not isinstance(self.num_gpus, (int, float)):
            raise ValueError("client-resources.num-gpus must be a number (int/float).")


@dataclass
class SimulationInitArgs:
    """Initialization arguments for the simulation."""

    num_cpus: int | None = None
    num_gpus: int | None = None
    logging_level: str | None = None
    log_to_drive: bool | None = None

    def __post_init__(self) -> None:
        """Validate initialization arguments."""
        if self.num_cpus is not None and not isinstance(self.num_cpus, int):
            raise ValueError("init-args.num-cpus must be an integer.")
        if self.num_gpus is not None and not isinstance(self.num_gpus, int):
            raise ValueError("init-args.num-gpus must be an integer.")
        if self.logging_level is not None and not isinstance(self.logging_level, str):
            raise ValueError("init-args.logging-level must be a string.")
        if self.log_to_drive is not None and not isinstance(self.log_to_drive, bool):
            raise ValueError("init-args.log-to-drive must be a boolean.")


@dataclass
class SimulationBackendConfig:
    """Backend configuration for the simulation."""

    client_resources: SimulationClientResources | None = None
    init_args: SimulationInitArgs | None = None
    name: str = DEFAULT_SIMULATION_BACKEND_NAME

    def __post_init__(self) -> None:
        """Validate backend configuration."""
        if not isinstance(self.name, str):
            raise ValueError("backend.name must be a string.")


@dataclass
class SuperLinkSimulationOptions:
    """Options for local simulation."""

    num_supernodes: int
    backend: SimulationBackendConfig | None = None

    def __post_init__(self) -> None:
        """Validate simulation options."""
        if not isinstance(self.num_supernodes, int):
            raise ValueError(
                "Invalid simulation options: num-supernodes must be an integer."
            )


@dataclass
class SuperLinkConnection:
    """SuperLink connection configuration for CLI commands."""

    name: str
    _address: str | None = None
    _root_certificates: str | None = None
    _insecure: bool | None = None
    _enable_account_auth: bool | None = None
    _federation: str | None = None
    _options: SuperLinkSimulationOptions | None = None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        name: str,
        address: str | None = None,
        root_certificates: str | None = None,
        insecure: bool | None = None,
        enable_account_auth: bool | None = None,
        federation: str | None = None,
        options: SuperLinkSimulationOptions | None = None,
    ) -> None:
        self.name = name
        self._address = address
        self._root_certificates = root_certificates
        self._insecure = insecure
        self._enable_account_auth = enable_account_auth
        self._federation = federation
        self._options = options

        self.__post_init__()

    @property
    def address(self) -> str:
        """Return the address."""
        if self._address is None:
            raise ValueError("SuperLinkConnection.address is None")
        return self._address

    @property
    def root_certificates(self) -> str:
        """Return the root certificates."""
        if self._root_certificates is None:
            raise ValueError("SuperLinkConnection.root_certificates is None")
        return self._root_certificates

    @property
    def insecure(self) -> bool:
        """Return the insecure flag."""
        if self._insecure is None:
            raise ValueError("SuperLinkConnection.insecure is None")
        return self._insecure

    @property
    def enable_account_auth(self) -> bool:
        """Return the enable_account_auth flag."""
        if self._enable_account_auth is None:
            raise ValueError("SuperLinkConnection.enable_account_auth is None")
        return self._enable_account_auth

    @property
    def federation(self) -> str:
        """Return the federation."""
        if self._federation is None:
            raise ValueError("SuperLinkConnection.federation is None")
        return self._federation

    @property
    def options(self) -> SuperLinkSimulationOptions:
        """Return the simulation options."""
        if self._options is None:
            raise ValueError("SuperLinkConnection.options is None")
        return self._options

    def __post_init__(self) -> None:
        """Validate SuperLink connection configuration."""
        err_prefix = f"Invalid value for key '%s' in connection '{self.name}': "
        if self._address is not None and not isinstance(self._address, str):
            raise ValueError(
                err_prefix % SuperLinkConnectionTomlKey.ADDRESS
                + f"expected str, but got {type(self._address).__name__}."
            )
        if self._root_certificates is not None and not isinstance(
            self._root_certificates, str
        ):
            raise ValueError(
                err_prefix % SuperLinkConnectionTomlKey.ROOT_CERTIFICATES
                + f"expected str, but got {type(self._root_certificates).__name__}."
            )

        # Ensure root certificates path is absolute
        if self._root_certificates is not None:
            if not Path(self._root_certificates).is_absolute():
                raise ValueError(
                    err_prefix % SuperLinkConnectionTomlKey.ROOT_CERTIFICATES
                    + "expected absolute path, but got relative path "
                    f"'{self._root_certificates}'."
                )

        if self._insecure is not None and not isinstance(self._insecure, bool):
            raise ValueError(
                err_prefix % SuperLinkConnectionTomlKey.INSECURE
                + f"expected bool, but got {type(self._insecure).__name__}."
            )
        if self._enable_account_auth is not None and not isinstance(
            self._enable_account_auth, bool
        ):
            raise ValueError(
                err_prefix % SuperLinkConnectionTomlKey.ENABLE_ACCOUNT_AUTH
                + f"expected bool, but got {type(self._enable_account_auth).__name__}."
            )

        if self._federation is not None and not isinstance(self._federation, str):
            raise ValueError(
                err_prefix % SuperLinkConnectionTomlKey.FEDERATION
                + f"expected str, but got {type(self._federation).__name__}."
            )

        # The connection needs to have either an address or options (or both).
        if self._address is None and self._options is None:
            raise ValueError(
                "Invalid SuperLink connection format: "
                f"'{SuperLinkConnectionTomlKey.ADDRESS}' and/or "
                f"'{SuperLinkConnectionTomlKey.OPTIONS}' key "
                "need to be specified."
            )
