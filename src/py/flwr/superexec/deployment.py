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
"""Deployment engine executor."""

import hashlib
from logging import ERROR, INFO
from pathlib import Path
from typing import Optional

from typing_extensions import override

from flwr.common import Context, RecordSet
from flwr.common.constant import SERVERAPPIO_API_DEFAULT_ADDRESS, Status, SubStatus
from flwr.common.logger import log
from flwr.common.typing import Fab, RunStatus, UserConfig
from flwr.server.superlink.ffs import Ffs
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory

from .executor import Executor


class DeploymentEngine(Executor):
    """Deployment engine executor.

    Parameters
    ----------
    superlink: str (default: "0.0.0.0:9091")
        Address of the SuperLink to connect to.
    root_certificates: Optional[str] (default: None)
        Specifies the path to the PEM-encoded root certificate file for
        establishing secure HTTPS connections.
    flwr_dir: Optional[str] (default: None)
        The path containing installed Flower Apps.
    """

    def __init__(
        self,
        superlink: str = SERVERAPPIO_API_DEFAULT_ADDRESS,
        root_certificates: Optional[str] = None,
        flwr_dir: Optional[str] = None,
    ) -> None:
        self.superlink = superlink
        if root_certificates is None:
            self.root_certificates = None
            self.root_certificates_bytes = None
        else:
            self.root_certificates = root_certificates
            self.root_certificates_bytes = Path(root_certificates).read_bytes()
        self.flwr_dir = flwr_dir
        self.linkstate_factory: Optional[LinkStateFactory] = None
        self.ffs_factory: Optional[FfsFactory] = None

    @override
    def initialize(
        self, linkstate_factory: LinkStateFactory, ffs_factory: FfsFactory
    ) -> None:
        """Initialize the executor with the necessary factories."""
        self.linkstate_factory = linkstate_factory
        self.ffs_factory = ffs_factory

    @property
    def linkstate(self) -> LinkState:
        """Return the LinkState."""
        if self.linkstate_factory is None:
            raise RuntimeError("Executor is not initialized.")
        return self.linkstate_factory.state()

    @property
    def ffs(self) -> Ffs:
        """Return the Flower File Storage (FFS)."""
        if self.ffs_factory is None:
            raise RuntimeError("Executor is not initialized.")
        return self.ffs_factory.ffs()

    @override
    def set_config(
        self,
        config: UserConfig,
    ) -> None:
        """Set executor config arguments.

        Parameters
        ----------
        config : UserConfig
            A dictionary for configuration values.
            Supported configuration key/value pairs:
            - "superlink": str
                The address of the SuperLink ServerAppIo API.
            - "root-certificates": str
                The path to the root certificates.
            - "flwr-dir": str
                The path to the Flower directory.
        """
        if not config:
            return
        if superlink_address := config.get("superlink"):
            if not isinstance(superlink_address, str):
                raise ValueError("The `superlink` value should be of type `str`.")
            self.superlink = superlink_address
        if root_certificates := config.get("root-certificates"):
            if not isinstance(root_certificates, str):
                raise ValueError(
                    "The `root-certificates` value should be of type `str`."
                )
            self.root_certificates = root_certificates
            self.root_certificates_bytes = Path(str(root_certificates)).read_bytes()
        if flwr_dir := config.get("flwr-dir"):
            if not isinstance(flwr_dir, str):
                raise ValueError("The `flwr-dir` value should be of type `str`.")
            self.flwr_dir = str(flwr_dir)

    def _create_run(
        self,
        fab: Fab,
        override_config: UserConfig,
    ) -> int:
        fab_hash = self.ffs.put(fab.content, {})
        if fab_hash != fab.hash_str:
            raise RuntimeError(
                f"FAB ({fab.hash_str}) hash from request doesn't match contents"
            )

        run_id = self.linkstate.create_run(None, None, fab_hash, override_config)
        return run_id

    def _create_context(self, run_id: int) -> None:
        """Register a Context for a Run."""
        # Create an empty context for the Run
        context = Context(node_id=0, node_config={}, state=RecordSet(), run_config={})

        # Register the context at the LinkState
        self.linkstate.set_serverapp_context(run_id=run_id, context=context)

    @override
    def start_run(
        self,
        fab_file: bytes,
        override_config: UserConfig,
        federation_config: UserConfig,
    ) -> Optional[int]:
        """Start run using the Flower Deployment Engine."""
        run_id = None
        try:

            # Call SuperLink to create run
            run_id = self._create_run(
                Fab(hashlib.sha256(fab_file).hexdigest(), fab_file), override_config
            )

            # Register context for the Run
            self._create_context(run_id=run_id)
            log(INFO, "Created run %s", str(run_id))

            return run_id
        # pylint: disable-next=broad-except
        except Exception as e:
            log(ERROR, "Could not start run: %s", str(e))
            if run_id:
                run_status = RunStatus(Status.FINISHED, SubStatus.FAILED, str(e))
                self.linkstate.update_run_status(run_id, new_status=run_status)
            return None


executor = DeploymentEngine()
