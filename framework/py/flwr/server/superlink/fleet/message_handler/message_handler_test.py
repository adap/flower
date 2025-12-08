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
"""Fleet API message handler tests."""


from unittest.mock import MagicMock

from flwr.common import Metadata, RecordDict, now
from flwr.common.message import make_message
from flwr.common.serde import message_to_proto
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    PullMessagesRequest,
    PushMessagesRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611

from .message_handler import pull_messages, push_messages


def test_pull_messages() -> None:
    """Test pull_messages."""
    # Prepare
    request = PullMessagesRequest(node=Node(node_id=1234))
    state = MagicMock()
    store = MagicMock()

    # Execute
    pull_messages(request=request, state=state, store=store)

    # Assert
    state.create_node.assert_not_called()
    state.delete_node.assert_not_called()
    state.store_message_ins.assert_not_called()
    state.get_message_ins.assert_called_once()
    state.store_message_res.assert_not_called()
    state.get_message_res.assert_not_called()
    state.store_traffic.assert_not_called()


def test_push_messages() -> None:
    """Test push_messages."""
    # Prepare
    msg = make_message(
        content=RecordDict(),
        metadata=Metadata(
            run_id=123,
            message_id="",
            group_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=123,
            message_type="query",
        ),
    )

    request = PushMessagesRequest(messages_list=[message_to_proto(msg)])
    state = MagicMock()
    store = MagicMock()

    # Execute
    push_messages(request=request, state=state, store=store)

    # Assert
    state.create_node.assert_not_called()
    state.delete_node.assert_not_called()
    state.store_message_ins.assert_not_called()
    state.get_message_ins.assert_not_called()
    state.store_message_res.assert_called_once()
    state.get_message_res.assert_not_called()
    state.store_traffic.assert_called_once()
