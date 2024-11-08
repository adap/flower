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
"""Simulation engine executor."""


import hashlib
from logging import ERROR, INFO
from typing import Optional

from typing_extensions import override

from flwr.common import ConfigsRecord, Context, RecordSet
from flwr.common.logger import log
from flwr.common.typing import Fab, UserConfig
from flwr.server.superlink.ffs import Ffs
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory

from .executor import Executor


class SimulationEngine(Executor):
    """Simulation engine executor.

    Parameters
    ----------
    num_supernodes: Opitonal[str] (default: None)
        Total number of nodes to involve in the simulation.
    """

    def __init__(
        self,
        num_supernodes: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.num_supernodes = num_supernodes
        self.verbose = verbose
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
            - "num-supernodes": int
                Number of nodes to register for the simulation.
            - "verbose": bool
                Set verbosity of logs.
        """
        if num_supernodes := config.get("num-supernodes"):
            if not isinstance(num_supernodes, int):
                raise ValueError("The `num-supernodes` value should be of type `int`.")
            self.num_supernodes = num_supernodes
        elif self.num_supernodes is None:
            log(
                ERROR,
                "To start a run with the simulation plugin, please specify "
                "the number of SuperNodes. This can be done by using the "
                "`--executor-config` argument when launching the SuperExec.",
            )
            raise ValueError(
                "`num-supernodes` must not be `None`, it must be a valid "
                "positive integer."
            )

        if verbose := config.get("verbose"):
            if not isinstance(verbose, bool):
                raise ValueError(
                    "The `verbose` value must be a string `true` or `false`."
                )
            self.verbose = verbose

    # pylint: disable=too-many-locals
    @override
    def start_run(
        self,
        fab_file: bytes,
        override_config: UserConfig,
        federation_options: ConfigsRecord,
    ) -> Optional[int]:
        """Start run using the Flower Simulation Engine."""
        try:
            # Create run
            fab = Fab(hashlib.sha256(fab_file).hexdigest(), fab_file)
            fab_hash = self.ffs.put(fab.content, {})
            if fab_hash != fab.hash_str:
                raise RuntimeError(
                    f"FAB ({fab.hash_str}) hash from request doesn't match contents"
                )

            run_id = self.linkstate.create_run(
                None, None, fab_hash, override_config, federation_options
            )

            # Create an empty context for the Run
            context = Context(
                run_id=run_id,
                node_id=0,
                node_config={},
                state=RecordSet(),
                run_config={},
            )

            # Register the context at the LinkState
            self.linkstate.set_serverapp_context(run_id=run_id, context=context)

            log(INFO, "Created run %s", str(run_id))

            return run_id

        # pylint: disable-next=broad-except
        except Exception as e:
            log(ERROR, "Could not start run: %s", str(e))
            return None


executor = SimulationEngine()
