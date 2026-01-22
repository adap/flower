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
"""Test for utility functions."""
# pylint: disable=invalid-name, disable=R0904

import unittest
from copy import deepcopy

from parameterized import parameterized

from flwr.app.error import Error
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.common.serde import message_from_proto
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.server.superlink.linkstate.utils import dict_to_message, message_to_dict


class SqliteStateTest(unittest.TestCase):
    """Test utilitiy functions."""

    @parameterized.expand([(False,), (True,)])  # type: ignore
    def test_message_to_dict_and_back(self, has_error: bool) -> None:
        """Check if all required keys are included in return value."""
        # Prepare
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=123, run_id=456
            )
        )
        if has_error:
            msg.__dict__["_content"] = None
            msg.__dict__["_error"] = Error(0)

        expected_keys = [
            "message_id",
            "group_id",
            "run_id",
            "src_node_id",
            "dst_node_id",
            "reply_to_message_id",
            "created_at",
            "delivered_at",
            "ttl",
            "message_type",
            "content",
            "error",
        ]

        # Execute
        result = message_to_dict(deepcopy(msg))

        # Assert
        for key in expected_keys:
            assert key in result

        # Execute
        res_msg = dict_to_message(result)

        # Assert
        if has_error:
            assert res_msg.error.code == msg.error.code
        else:
            assert res_msg.content == msg.content
        assert res_msg.metadata == msg.metadata


if __name__ == "__main__":
    unittest.main(verbosity=2)
