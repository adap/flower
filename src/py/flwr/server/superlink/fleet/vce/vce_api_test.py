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
"""Test Fleet Simulation Engine API."""


import asyncio
import threading
import time
from itertools import cycle
from json import JSONDecodeError
from math import pi
from pathlib import Path
from time import sleep
from typing import Dict, Optional, Set, Tuple
from unittest import TestCase
from uuid import UUID

from flwr.client.client_app import LoadClientAppError
from flwr.common import (
    DEFAULT_TTL,
    GetPropertiesIns,
    Message,
    MessageTypeLegacy,
    Metadata,
)
from flwr.common.recordset_compat import getpropertiesins_to_recordset
from flwr.common.serde import message_from_taskres, message_to_taskins
from flwr.common.typing import Run
from flwr.server.superlink.fleet.vce.vce_api import (
    NodeToPartitionMapping,
    _register_nodes,
    start_vce,
)
from flwr.server.superlink.state import InMemoryState, StateFactory


def terminate_simulation(f_stop: asyncio.Event, sleep_duration: int) -> None:
    """Set event to terminate Simulation Engine after `sleep_duration` seconds."""
    sleep(sleep_duration)
    f_stop.set()


def init_state_factory_nodes_mapping(
    num_nodes: int,
    num_messages: int,
) -> Tuple[StateFactory, NodeToPartitionMapping, Dict[UUID, float]]:
    """Instatiate StateFactory, register nodes and pre-insert messages in the state."""
    # Register a state and a run_id in it
    run_id = 1234
    state_factory = StateFactory(":flwr-in-memory-state:")

    # Register a few nodes
    nodes_mapping = _register_nodes(num_nodes=num_nodes, state_factory=state_factory)

    expected_results = register_messages_into_state(
        state_factory=state_factory,
        nodes_mapping=nodes_mapping,
        run_id=run_id,
        num_messages=num_messages,
    )
    return state_factory, nodes_mapping, expected_results


# pylint: disable=too-many-locals
def register_messages_into_state(
    state_factory: StateFactory,
    nodes_mapping: NodeToPartitionMapping,
    run_id: int,
    num_messages: int,
) -> Dict[UUID, float]:
    """Register `num_messages` into the state factory."""
    state: InMemoryState = state_factory.state()  # type: ignore
    state.run_ids[run_id] = Run(
        run_id=run_id, fab_id="Mock/mock", fab_version="v1.0.0", override_config={}
    )
    # Artificially add TaskIns to state so they can be processed
    # by the Simulation Engine logic
    nodes_cycle = cycle(nodes_mapping.keys())  # we have more messages than supernodes
    task_ids: Set[UUID] = set()  # so we can retrieve them later
    expected_results = {}
    for i in range(num_messages):
        dst_node_id = next(nodes_cycle)
        # Construct a Message
        mult_factor = 2024 + i
        getproperties_ins = GetPropertiesIns(config={"factor": mult_factor})
        recordset = getpropertiesins_to_recordset(getproperties_ins)
        message = Message(
            content=recordset,
            metadata=Metadata(
                run_id=run_id,
                message_id="",
                group_id="",
                src_node_id=0,
                dst_node_id=dst_node_id,  # indicate destination node
                reply_to_message="",
                ttl=DEFAULT_TTL,
                message_type=MessageTypeLegacy.GET_PROPERTIES,
            ),
        )
        # Convert Message to TaskIns
        taskins = message_to_taskins(message)
        # Normally recorded by the driver servicer
        # but since we don't have one in this test, we do this manually
        taskins.task.pushed_at = time.time()
        # Instert in state
        task_id = state.store_task_ins(taskins)
        if task_id:
            # Add to UUID set
            task_ids.add(task_id)
            # Store expected output for check later on
            expected_results[task_id] = mult_factor * pi

    return expected_results


def _autoresolve_app_dir(rel_client_app_dir: str = "backend") -> str:
    """Correctly resolve working directory for the app."""
    file_path = Path(__file__)
    app_dir = Path.cwd()
    rel_app_dir = file_path.relative_to(app_dir)

    # Susbtract lats element and append "backend/test" (wher the client module is.)
    return str(rel_app_dir.parent / rel_client_app_dir)


# pylint: disable=too-many-arguments
def start_and_shutdown(
    backend: str = "ray",
    client_app_attr: str = "raybackend_test:client_app",
    app_dir: str = "",
    num_supernodes: Optional[int] = None,
    state_factory: Optional[StateFactory] = None,
    nodes_mapping: Optional[NodeToPartitionMapping] = None,
    duration: int = 0,
    backend_config: str = "{}",
) -> None:
    """Start Simulation Engine and terminate after specified number of seconds.

    Some tests need to be terminated by triggering externally an asyncio.Event. This
    is enabled when passing `duration`>0.
    """
    f_stop = asyncio.Event()

    if duration:

        # Setup thread that will set the f_stop event, triggering the termination of all
        # asyncio logic in the Simulation Engine. It will also terminate the Backend.
        termination_th = threading.Thread(
            target=terminate_simulation, args=(f_stop, duration)
        )
        termination_th.start()

    # Resolve working directory if not passed
    if not app_dir:
        app_dir = _autoresolve_app_dir()

    start_vce(
        num_supernodes=num_supernodes,
        client_app_attr=client_app_attr,
        backend_name=backend,
        backend_config_json_stream=backend_config,
        state_factory=state_factory,
        app_dir=app_dir,
        f_stop=f_stop,
        existing_nodes_mapping=nodes_mapping,
    )

    if duration:
        termination_th.join()


class TestFleetSimulationEngineRayBackend(TestCase):
    """A basic class that enables testing functionalities."""

    def test_erroneous_no_supernodes_client_mapping(self) -> None:
        """Test with unset arguments."""
        with self.assertRaises(ValueError):
            start_and_shutdown(duration=2)

    def test_erroneous_client_app_attr(self) -> None:
        """Tests attempt to load a ClientApp that can't be found."""
        num_messages = 7
        num_nodes = 59

        state_factory, nodes_mapping, _ = init_state_factory_nodes_mapping(
            num_nodes=num_nodes, num_messages=num_messages
        )
        with self.assertRaises(LoadClientAppError):
            start_and_shutdown(
                client_app_attr="totally_fictitious_app:client",
                state_factory=state_factory,
                nodes_mapping=nodes_mapping,
            )

    def test_erroneous_backend_config(self) -> None:
        """Backend Config should be a JSON stream."""
        with self.assertRaises(JSONDecodeError):
            start_and_shutdown(num_supernodes=50, backend_config="not a proper config")

    def test_with_nonexistent_backend(self) -> None:
        """Test specifying a backend that does not exist."""
        with self.assertRaises(KeyError):
            start_and_shutdown(num_supernodes=50, backend="this-backend-does-not-exist")

    def test_erroneous_arguments_num_supernodes_and_existing_mapping(self) -> None:
        """Test ValueError if a node mapping is passed but also num_supernodes.

        Passing `num_supernodes` does nothing since we assume that if a node mapping
        is supplied, nodes have been registered externally already. Therefore passing
        `num_supernodes` might give the impression that that many nodes will be
        registered. We don't do that since a mapping already exists.
        """
        with self.assertRaises(ValueError):
            start_and_shutdown(num_supernodes=50, nodes_mapping={0: 1})

    def test_erroneous_arguments_existing_mapping_but_no_state_factory(self) -> None:
        """Test ValueError if a node mapping is passed but no state.

        Passing a node mapping indicates that (externally) nodes have registered with a
        state factory. Therefore, that state factory should be passed too.
        """
        with self.assertRaises(ValueError):
            start_and_shutdown(nodes_mapping={0: 1})

    def test_start_and_shutdown(self) -> None:
        """Start Simulation Engine Fleet and terminate it."""
        start_and_shutdown(num_supernodes=50, duration=10)

    # pylint: disable=too-many-locals
    def test_start_and_shutdown_with_tasks_in_state(self) -> None:
        """Run Simulation Engine with some TasksIns in State.

        This test creates a few nodes and submits a few messages that need to be
        executed by the Backend. In order for that to happen the asyncio
        producer/consumer logic must function. This also severs to evaluate a valid
        ClientApp.
        """
        num_messages = 229
        num_nodes = 59

        state_factory, nodes_mapping, expected_results = (
            init_state_factory_nodes_mapping(
                num_nodes=num_nodes, num_messages=num_messages
            )
        )

        # Run
        start_and_shutdown(
            state_factory=state_factory, nodes_mapping=nodes_mapping, duration=10
        )

        # Get all TaskRes
        state = state_factory.state()
        task_ids = set(expected_results.keys())
        task_res_list = state.get_task_res(task_ids=task_ids, limit=len(task_ids))

        # Check results by first converting to Message
        for task_res in task_res_list:

            message = message_from_taskres(task_res)

            # Verify message content is as expected
            content = message.content
            assert (
                content.configs_records["getpropertiesres.properties"]["result"]
                == expected_results[UUID(task_res.task.ancestry[0])]
            )
