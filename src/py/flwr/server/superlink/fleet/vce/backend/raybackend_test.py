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
"""Test for Ray backend for the Fleet API using the Simulation Engine."""

from math import pi
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
from unittest import TestCase

import ray

from flwr.client import Client, NumPyClient
from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.client.node_state import NodeState
from flwr.common import (
    DEFAULT_TTL,
    Config,
    Context,
    GetPropertiesIns,
    Message,
    MessageTypeLegacy,
    Metadata,
    Scalar,
)
from flwr.common.constant import PARTITION_ID_KEY
from flwr.common.object_ref import load_app
from flwr.common.recordset_compat import getpropertiesins_to_recordset
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.server.superlink.fleet.vce.backend.raybackend import RayBackend


class DummyClient(NumPyClient):
    """A dummy NumPyClient for tests."""

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Return properties by doing a simple calculation."""
        result = float(config["factor"]) * pi

        return {"result": result}


def get_dummy_client(context: Context) -> Client:  # pylint: disable=unused-argument
    """Return a DummyClient converted to Client type."""
    return DummyClient().to_client()


def _load_app() -> ClientApp:
    return ClientApp(client_fn=get_dummy_client)


client_app = ClientApp(
    client_fn=get_dummy_client,
)


def _load_from_module(client_app_module_name: str) -> Callable[[], ClientApp]:
    def _load_app() -> ClientApp:
        app = load_app(client_app_module_name, LoadClientAppError)

        if not isinstance(app, ClientApp):
            raise LoadClientAppError(
                f"Attribute {client_app_module_name} is not of type {ClientApp}",
            ) from None

        return app

    return _load_app


def backend_build_process_and_termination(
    backend: RayBackend,
    process_args: Optional[Tuple[Callable[[], ClientApp], Message, Context]] = None,
) -> Union[Tuple[Message, Context], None]:
    """Build, process job and terminate RayBackend."""
    backend.build()
    to_return = None

    if process_args:
        to_return = backend.process_message(*process_args)

    backend.terminate()

    return to_return


def _create_message_and_context() -> Tuple[Message, Context, float]:

    # Construct a Message
    mult_factor = 2024
    run_id = 0
    getproperties_ins = GetPropertiesIns(config={"factor": mult_factor})
    recordset = getpropertiesins_to_recordset(getproperties_ins)
    message = Message(
        content=recordset,
        metadata=Metadata(
            run_id=run_id,
            message_id="",
            group_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message="",
            ttl=DEFAULT_TTL,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
        ),
    )

    # Construct NodeState and retrieve context
    node_state = NodeState(node_id=run_id, node_config={PARTITION_ID_KEY: str(0)})
    node_state.register_context(run_id=run_id)
    context = node_state.retrieve_context(run_id=run_id)

    # Expected output
    expected_output = pi * mult_factor

    return message, context, expected_output


class TestRayBackend(TestCase):
    """A basic class that allows runnig multliple tests."""

    def doCleanups(self) -> None:
        """Ensure Ray has shutdown."""
        if ray.is_initialized():
            ray.shutdown()

    def test_backend_creation_and_termination(self) -> None:
        """Test creation of RayBackend and its termination."""
        backend = RayBackend(backend_config={}, work_dir="")
        backend_build_process_and_termination(backend=backend, process_args=None)

    def test_backend_creation_submit_and_termination(
        self,
        client_app_loader: Callable[[], ClientApp] = _load_app,
        workdir: str = "",
    ) -> None:
        """Test submitting a message to a given ClientApp."""
        backend = RayBackend(backend_config={}, work_dir=workdir)

        # Define ClientApp
        client_app_callable = client_app_loader

        message, context, expected_output = _create_message_and_context()

        res = backend_build_process_and_termination(
            backend=backend, process_args=(client_app_callable, message, context)
        )

        if res is None:
            raise AssertionError("This shouldn't happen")

        out_mssg, updated_context = res

        # Verify message content is as expected
        content = out_mssg.content
        assert (
            content.configs_records["getpropertiesres.properties"]["result"]
            == expected_output
        )

    def test_backend_creation_submit_and_termination_non_existing_client_app(
        self,
    ) -> None:
        """Testing with ClientApp module that does not exist."""
        with self.assertRaises(LoadClientAppError):
            self.test_backend_creation_submit_and_termination(
                client_app_loader=_load_from_module("a_non_existing_module:app")
            )

    def test_backend_creation_submit_and_termination_existing_client_app(
        self,
    ) -> None:
        """Testing with ClientApp module that exist."""
        # Resolve what should be the workdir to pass upon Backend initialisation
        file_path = Path(__file__)
        working_dir = Path.cwd()
        rel_workdir = file_path.relative_to(working_dir)

        # Susbtract last element
        rel_workdir_str = str(rel_workdir.parent)

        self.test_backend_creation_submit_and_termination(
            client_app_loader=_load_from_module("raybackend_test:client_app"),
            workdir=rel_workdir_str,
        )

    def test_backend_creation_submit_and_termination_existing_client_app_unsetworkdir(
        self,
    ) -> None:
        """Testing with ClientApp module that exist but the passed workdir does not."""
        with self.assertRaises(ValueError):
            self.test_backend_creation_submit_and_termination(
                client_app_loader=_load_from_module("raybackend_test:client_app"),
                workdir="/?&%$^#%@$!",
            )

    def test_backend_creation_with_init_arguments(self) -> None:
        """Testing whether init args are properly parsed to Ray."""
        backend_config_4: BackendConfig = {
            "init_args": {"num_cpus": 4},
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        }

        backend_config_2: BackendConfig = {
            "init_args": {"num_cpus": 2},
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        }

        RayBackend(
            backend_config=backend_config_4,
            work_dir="",
        )
        nodes = ray.nodes()

        assert nodes[0]["Resources"]["CPU"] == backend_config_4["init_args"]["num_cpus"]

        ray.shutdown()

        RayBackend(
            backend_config=backend_config_2,
            work_dir="",
        )
        nodes = ray.nodes()

        assert nodes[0]["Resources"]["CPU"] == backend_config_2["init_args"]["num_cpus"]
