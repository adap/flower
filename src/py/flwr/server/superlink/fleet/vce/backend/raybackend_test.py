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

import asyncio
from math import pi
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
from unittest import IsolatedAsyncioTestCase

import ray

from flwr.client import Client, NumPyClient
from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.common import (
    DEFAULT_TTL,
    Config,
    ConfigsRecord,
    Context,
    GetPropertiesIns,
    Message,
    MessageTypeLegacy,
    Metadata,
    RecordSet,
    Scalar,
)
from flwr.common.constant import ErrorCode
from flwr.common.object_ref import load_app
from flwr.common.recordset_compat import getpropertiesins_to_recordset
from flwr.server.superlink.fleet.vce.backend.raybackend import RayBackend


class DummyClient(NumPyClient):
    """A dummy NumPyClient for tests."""

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Return properties by doing a simple calculation."""
        result = float(config["factor"]) * pi

        # store something in context
        self.context.state.configs_records["result"] = ConfigsRecord({"result": result})
        return {"result": result}


def get_dummy_client(cid: str) -> Client:  # pylint: disable=unused-argument
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


async def backend_build_process_and_termination(
    backend: RayBackend,
    process_args: Optional[Tuple[Callable[[], ClientApp], Message, Context]] = None,
) -> Union[Tuple[Message, Context], None]:
    """Build, process job and terminate RayBackend."""
    await backend.build()
    to_return = None

    if process_args:
        to_return = await backend.process_message(*process_args)

    await backend.terminate()

    return to_return


def _create_message_and_context() -> Tuple[Message, Context, float]:

    # Construct a Message
    mult_factor = 2024
    getproperties_ins = GetPropertiesIns(config={"factor": mult_factor})
    recordset = getpropertiesins_to_recordset(getproperties_ins)
    message = Message(
        content=recordset,
        metadata=Metadata(
            run_id=0,
            message_id="",
            group_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message="",
            ttl=DEFAULT_TTL,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
        ),
    )

    # Construct emtpy Context
    context = Context(state=RecordSet())

    # Expected output
    expected_output = pi * mult_factor

    return message, context, expected_output


class AsyncTestRayBackend(IsolatedAsyncioTestCase):
    """A basic class that allows runnig multliple asyncio tests."""

    async def on_cleanup(self) -> None:
        """Ensure Ray has shutdown."""
        if ray.is_initialized():
            ray.shutdown()

    def test_backend_creation_and_termination(self) -> None:
        """Test creation of RayBackend and its termination."""
        backend = RayBackend(backend_config={}, work_dir="")
        asyncio.run(
            backend_build_process_and_termination(backend=backend, process_args=None)
        )

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

        res = asyncio.run(
            backend_build_process_and_termination(
                backend=backend, process_args=(client_app_callable, message, context)
            )
        )

        if res is None:
            raise AssertionError("This shouldn't happen")

        out_mssg, updated_context = res

        if out_mssg.has_error():
            if out_mssg.error.code == ErrorCode.LOAD_CLIENT_APP_EXCEPTION:
                raise LoadClientAppError()

        # Verify message content is as expected
        content = out_mssg.content
        assert (
            content.configs_records["getpropertiesres.properties"]["result"]
            == expected_output
        )

        # Verify context is correct
        obtained_result_in_context = updated_context.state.configs_records["result"][
            "result"
        ]
        assert obtained_result_in_context == expected_output

    def test_backend_creation_submit_and_termination_non_existing_client_app(
        self,
    ) -> None:
        """Testing with ClientApp module that does not exist."""
        with self.assertRaises(LoadClientAppError):
            self.test_backend_creation_submit_and_termination(
                client_app_loader=_load_from_module("a_non_existing_module:app")
            )
        self.addAsyncCleanup(self.on_cleanup)

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
        self.addAsyncCleanup(self.on_cleanup)
