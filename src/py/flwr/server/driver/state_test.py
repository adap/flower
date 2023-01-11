# Copyright 2022 Adap GmbH. All Rights Reserved.
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
"""DriverState tests."""


from uuid import uuid4

from flwr.server.driver.state import DriverState


def test_get_task_ins_empty() -> None:
    """."""

    # Prepare
    state = DriverState()

    # Execute
    task_ins_set = state.get_task_ins(
        node_id=None,
        limit=10,
    )

    # Assert
    assert not task_ins_set


def test_get_task_res_empty() -> None:
    """."""

    # Prepare
    state = DriverState()

    # Execute
    task_ins_set = state.get_task_res(
        node_id=123,
        task_ids={uuid4()},
        limit=10,
    )

    # Assert
    assert not task_ins_set
