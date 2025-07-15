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
"""Test the ClientAppIo API servicer."""


import unittest
from unittest.mock import Mock

from flwr.common import Context, typing
from flwr.common.message import make_message
from flwr.common.serde import fab_to_proto, message_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.proto.appio_pb2 import (  # pylint:disable=E0611
    PullAppInputsResponse,
    PullAppMessagesResponse,
    PushAppOutputsResponse,
)
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint:disable=E0611
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint:disable=E0611
from flwr.supernode.runtime.run_clientapp import (
    pull_clientappinputs,
    push_clientappoutputs,
)

from .clientappio_servicer import ClientAppIoServicer


class TestClientAppIoServicer(unittest.TestCase):
    """Tests for `ClientAppIoServicer` class."""

    def setUp(self) -> None:
        """Initialize."""
        self.servicer = ClientAppIoServicer(Mock(), Mock(), Mock())
        self.maker = RecordMaker()
        self.mock_stub = Mock()

    def test_pull_clientapp_inputs(self) -> None:
        """Test pulling messages from SuperNode."""
        # Prepare
        mock_message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(3, 2, 1),
        )
        mock_fab = typing.Fab(
            hash_str="abc123#$%",
            content=b"\xf3\xf5\xf8\x98",
        )
        mock_response = PullAppInputsResponse(
            context=ProtoContext(node_id=123),
            run=ProtoRun(run_id=61016, fab_id="mock/mock", fab_version="v1.0.0"),
            fab=fab_to_proto(mock_fab),
        )
        self.mock_stub.PullMessage.return_value = PullAppMessagesResponse(
            messages_list=[message_to_proto(mock_message)]
        )
        self.mock_stub.PullClientAppInputs.return_value = mock_response

        # Execute
        message, context, run, fab = pull_clientappinputs(self.mock_stub, token="abc")

        # Assert
        self.mock_stub.PullClientAppInputs.assert_called_once()
        self.assertEqual(len(message.content.array_records), 3)
        self.assertEqual(len(message.content.metric_records), 2)
        self.assertEqual(len(message.content.config_records), 1)
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
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(2, 2, 1),
        )
        context = Context(
            run_id=1,
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recorddict(2, 2, 1),
            run_config={"runconfig1": 6.1},
        )
        mock_response = PushAppOutputsResponse()
        self.mock_stub.PushClientAppOutputs.return_value = mock_response

        # Execute
        _ = push_clientappoutputs(
            stub=self.mock_stub, token="abc", message=message, context=context
        )

        # Assert
        self.mock_stub.PushClientAppOutputs.assert_called_once()
        self.mock_stub.PushMessage.assert_called_once()
