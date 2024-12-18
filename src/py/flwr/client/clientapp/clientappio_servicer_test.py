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

from flwr.client.clientapp.app import get_token, pull_message, push_message
from flwr.common import Context, Message, typing
from flwr.common.constant import RUN_ID_NUM_BYTES
from flwr.common.serde import (
    clientappstatus_from_proto,
    clientappstatus_to_proto,
    fab_to_proto,
    message_to_proto,
)
from flwr.common.serde_test import RecordMaker

# pylint:disable=E0611
from flwr.proto.clientappio_pb2 import (
    GetTokenResponse,
    PullClientAppInputsResponse,
    PushClientAppOutputsResponse,
)
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.run_pb2 import Run as ProtoRun
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes

from .clientappio_servicer import ClientAppInputs, ClientAppIoServicer, ClientAppOutputs


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
            run_id=1,
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
            pending_at="2021-01-01T00:00:00Z",
            starting_at="",
            running_at="",
            finished_at="",
            status=typing.RunStatus(status="pending", sub_status="", details=""),
        )
        fab = typing.Fab(
            hash_str="abc123#$%",
            content=b"\xf3\xf5\xf8\x98",
        )

        client_input = ClientAppInputs(message, context, run, fab, 1)
        client_output = ClientAppOutputs(message, context)

        # Execute and assert
        # - when ClientAppInputs is not None, ClientAppOutputs is None
        with self.assertRaises(ValueError):
            self.servicer.clientapp_input = client_input
            self.servicer.clientapp_output = None
            self.servicer.set_inputs(client_input, token_returned=True)

        # Execute and assert
        # - when ClientAppInputs is None, ClientAppOutputs is not None
        with self.assertRaises(ValueError):
            self.servicer.clientapp_input = None
            self.servicer.clientapp_output = client_output
            self.servicer.set_inputs(client_input, token_returned=True)

        # Execute and assert
        # - when ClientAppInputs and ClientAppOutputs is not None
        with self.assertRaises(ValueError):
            self.servicer.clientapp_input = client_input
            self.servicer.clientapp_output = client_output
            self.servicer.set_inputs(client_input, token_returned=True)

        # Execute and assert
        # - when ClientAppInputs is set at .clientapp_input
        self.servicer.clientapp_input = None
        self.servicer.clientapp_output = None
        self.servicer.set_inputs(client_input, token_returned=True)
        assert client_input == self.servicer.clientapp_input

    def test_get_outputs(self) -> None:
        """Test getting ClientApp outputs."""
        # Prepare
        message = Message(
            metadata=self.maker.metadata(),
            content=self.maker.recordset(2, 2, 1),
        )
        context = Context(
            run_id=1,
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recordset(2, 2, 1),
            run_config={"runconfig1": 6.1},
        )
        client_output = ClientAppOutputs(message, context)

        # Execute and assert - when `ClientAppOutputs` is None
        self.servicer.clientapp_output = None
        with self.assertRaises(ValueError):
            # `ClientAppOutputs` should not be None
            _ = self.servicer.get_outputs()

        # Execute and assert - when `ClientAppOutputs` is not None
        self.servicer.clientapp_output = client_output
        output = self.servicer.get_outputs()
        assert isinstance(output, ClientAppOutputs)
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
        mock_fab = typing.Fab(
            hash_str="abc123#$%",
            content=b"\xf3\xf5\xf8\x98",
        )
        mock_response = PullClientAppInputsResponse(
            message=message_to_proto(mock_message),
            context=ProtoContext(node_id=123),
            run=ProtoRun(run_id=61016, fab_id="mock/mock", fab_version="v1.0.0"),
            fab=fab_to_proto(mock_fab),
        )
        self.mock_stub.PullClientAppInputs.return_value = mock_response

        # Execute
        message, context, run, fab = pull_message(self.mock_stub, token=456)

        # Assert
        self.mock_stub.PullClientAppInputs.assert_called_once()
        self.assertEqual(len(message.content.parameters_records), 3)
        self.assertEqual(len(message.content.metrics_records), 2)
        self.assertEqual(len(message.content.configs_records), 1)
        self.assertEqual(context.node_id, 123)
        self.assertEqual(run.run_id, 61016)
        self.assertEqual(run.fab_id, "mock/mock")
        self.assertEqual(run.fab_version, "v1.0.0")
        if fab:
            self.assertEqual(fab.hash_str, mock_fab.hash_str)
            self.assertEqual(fab.content, mock_fab.content)

    def test_push_clientapp_outputs(self) -> None:
        """Test pushing messages to SuperNode."""
        # Prepare
        message = Message(
            metadata=self.maker.metadata(),
            content=self.maker.recordset(2, 2, 1),
        )
        context = Context(
            run_id=1,
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

    def test_get_token(self) -> None:
        """Test getting a token from SuperNode."""
        # Prepare
        token: int = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
        mock_response = GetTokenResponse(token=token)
        self.mock_stub.GetToken.return_value = mock_response

        # Execute
        res = get_token(stub=self.mock_stub)

        # Assert
        self.mock_stub.GetToken.assert_called_once()
        self.assertEqual(res, token)
