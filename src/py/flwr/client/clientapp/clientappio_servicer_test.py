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
from unittest.mock import Mock, patch

from flwr.client.clientapp.app import pull_message, push_message
from flwr.common import Context, Message, typing
from flwr.common.serde import (
    clientappstatus_from_proto,
    clientappstatus_to_proto,
    message_to_proto,
)
from flwr.common.serde_test import RecordMaker

# pylint:disable=E0611
from flwr.proto.clientappio_pb2 import (
    PullClientAppInputsResponse,
    PushClientAppOutputsResponse,
)
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.run_pb2 import Run as ProtoRun

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
        self.mock_stub = Mock()
        self.patcher = patch(
            "flwr.client.clientapp.app.ClientAppIoStub", return_value=self.mock_stub
        )
        self.patcher.start()

    def tearDown(self) -> None:
        """Cleanup."""
        self.patcher.stop()

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

    def test_pull_clientapp_inputs(self) -> None:
        """Test pulling messages from SuperNode."""
        # Prepare
        mock_message = Message(
            metadata=self.maker.metadata(),
            content=self.maker.recordset(3, 2, 1),
        )
        mock_response = PullClientAppInputsResponse(
            message=message_to_proto(mock_message),
            context=ProtoContext(node_id=123),
            run=ProtoRun(run_id=61016, fab_id="mock/mock", fab_version="v1.0.0"),
        )
        self.mock_stub.PullClientAppInputs.return_value = mock_response

        # Execute
        message, context, run = pull_message(self.mock_stub, token=456)

        # Assert
        self.mock_stub.PullClientAppInputs.assert_called_once()
        self.assertEqual(len(message.content.parameters_records), 3)
        self.assertEqual(len(message.content.metrics_records), 2)
        self.assertEqual(len(message.content.configs_records), 1)
        self.assertEqual(context.node_id, 123)
        self.assertEqual(run.run_id, 61016)
        self.assertEqual(run.fab_id, "mock/mock")
        self.assertEqual(run.fab_version, "v1.0.0")

    def test_push_clientapp_outputs(self) -> None:
        """Test pushing messages to SuperNode."""
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
        code = typing.ClientAppOutputCode.SUCCESS
        status_proto = clientappstatus_to_proto(
            status=typing.ClientAppOutputStatus(code=code, message="SUCCESS"),
        )
        mock_response = PushClientAppOutputsResponse(status=status_proto)
        self.mock_stub.PushClientAppOutputs.return_value = mock_response

        # Execute
        res = push_message(
            stub=self.mock_stub, token=789, message=message, context=context
        )
        status = clientappstatus_from_proto(res.status)

        # Assert
        self.mock_stub.PushClientAppOutputs.assert_called_once()
        self.assertEqual(status.message, "SUCCESS")
