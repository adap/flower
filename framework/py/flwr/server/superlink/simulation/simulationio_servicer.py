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
"""Compatibility servicer for simulation workloads using ServerAppIo."""

from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.serverappio.serverappio_servicer import ServerAppIoServicer
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory


class SimulationIoServicer(ServerAppIoServicer):
    """Compatibility alias for the ServerAppIo servicer."""

    def __init__(
        self,
        state_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory | None = None,
    ) -> None:
        super().__init__(
            state_factory=state_factory,
            ffs_factory=ffs_factory,
            objectstore_factory=objectstore_factory or ObjectStoreFactory(),
        )
