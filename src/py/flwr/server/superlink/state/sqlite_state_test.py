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
"""Test for utility functions."""
# pylint: disable=invalid-name, disable=R0904

import unittest

from flwr.server.superlink.state.sqlite_state import task_ins_to_dict
from flwr.server.superlink.state.state_test import create_task_ins


class SqliteStateTest(unittest.TestCase):
    """Test utilitiy functions."""

    def test_ins_res_to_dict(self) -> None:
        """Check if all required keys are included in return value."""
        # Prepare
        ins_res = create_task_ins(consumer_node_id=1, anonymous=True, run_id=0)
        expected_keys = [
            "task_id",
            "group_id",
            "run_id",
            "producer_anonymous",
            "producer_node_id",
            "consumer_anonymous",
            "consumer_node_id",
            "created_at",
            "delivered_at",
            "pushed_at",
            "ttl",
            "ancestry",
            "task_type",
            "recordset",
        ]

        # Execute
        result = task_ins_to_dict(ins_res)

        # Assert
        for key in expected_keys:
            assert key in result


if __name__ == "__main__":
    unittest.main(verbosity=2)