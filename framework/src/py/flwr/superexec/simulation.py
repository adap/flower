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

from flwr.cli.config_utils import get_fab_metadata
from flwr.common import ConfigsRecord, Context, RecordSet
from flwr.common.logger import log
from flwr.common.typing import Fab, UserConfig
from flwr.server.superlink.ffs import Ffs
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory

from .executor import Executor


class SimulationEngine(Executor):
    """Simulation engine executor."""

    def __init__(
        self,
    ) -> None:
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
        """Set executor config arguments."""

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
            # Check that num-supernodes is set
            if "num-supernodes" not in federation_options:
                raise ValueError(
                    "Federation options doesn't contain key `num-supernodes`."
                )

            # Create run
            fab = Fab(hashlib.sha256(fab_file).hexdigest(), fab_file)
            fab_hash = self.ffs.put(fab.content, {})
            if fab_hash != fab.hash_str:
                raise RuntimeError(
                    f"FAB ({fab.hash_str}) hash from request doesn't match contents"
                )
            fab_id, fab_version = get_fab_metadata(fab.content)

            run_id = self.linkstate.create_run(
                fab_id, fab_version, fab_hash, override_config, federation_options
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
