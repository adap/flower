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
"""Test the ClientAppIo API servicer."""

import unittest

from flwr.common import Context, Message, typing
from flwr.common.serde_test import RecordMaker

from .clientappio_servicer import (
    ClientAppIoInputs,
    ClientAppIoOutputs,
    ClientAppIoServicer,
)


class TestClientAppIoServicer(unittest.TestCase):
    """Tests for `ClientAppIoServicer` class."""

    def setUp(self) -> None:
        """Initialize."""
        self.servicer = ClientAppIoServicer()
        self.maker = RecordMaker()

    def tearDown(self) -> None:
        """Cleanup."""

    def test_set_inputs(self) -> None:
        """Test setting ClientApp inputs."""
        # Prepare
        message = Message(
            metadata=self.maker.metadata(),
            content=self.maker.recordset(2, 2, 1),
        )
        context = Context(
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recordset(2, 2, 1),
            run_config={"runconfig1": 6.1},
        )
        run = typing.Run(
            run_id=1,
            fab_id="lorem",
            fab_version="ipsum",
            fab_hash="dolor",
            override_config=self.maker.user_config(),
        )
        client_input = ClientAppIoInputs(message, context, run, 1)
        client_output = ClientAppIoOutputs(message, context)

        # Execute and assert
        # - when ClientAppIoInputs is not None, ClientAppIoOutputs is None
        with self.assertRaises(ValueError):
            self.servicer.clientapp_input = client_input
            self.servicer.clientapp_output = None
            self.servicer.set_inputs(client_input)

        # Execute and assert
        # - when ClientAppIoInputs is None, ClientAppIoOutputs is not None
        with self.assertRaises(ValueError):
            self.servicer.clientapp_input = None
            self.servicer.clientapp_output = client_output
            self.servicer.set_inputs(client_input)

        # Execute and assert
        # - when ClientAppIoInputs and ClientAppIoOutputs is not None
        with self.assertRaises(ValueError):
            self.servicer.clientapp_input = client_input
            self.servicer.clientapp_output = client_output
            self.servicer.set_inputs(client_input)

        # Execute and assert
        # - when ClientAppIoInputs is set at .clientapp_input
        self.servicer.clientapp_input = None
        self.servicer.clientapp_output = None
        self.servicer.set_inputs(client_input)
        assert client_input == self.servicer.clientapp_input

    def test_get_outputs(self) -> None:
        """Test getting ClientApp outputs."""
        # Prepare
        message = Message(
            metadata=self.maker.metadata(),
            content=self.maker.recordset(2, 2, 1),
        )
        context = Context(
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recordset(2, 2, 1),
            run_config={"runconfig1": 6.1},
        )
        client_output = ClientAppIoOutputs(message, context)

        # Execute and assert - when `ClientAppIoOutputs` is None
        self.servicer.clientapp_output = None
        with self.assertRaises(ValueError):
            # `ClientAppIoOutputs` should not be None
            _ = self.servicer.get_outputs()

        # Execute and assert - when `ClientAppIoOutputs` is not None
        self.servicer.clientapp_output = client_output
        output = self.servicer.get_outputs()
        assert isinstance(output, ClientAppIoOutputs)
        assert output == client_output
        assert self.servicer.clientapp_input is None
        assert self.servicer.clientapp_output is None
