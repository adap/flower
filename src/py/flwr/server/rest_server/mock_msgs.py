# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Mock Messages for API server."""
import uuid
from typing import Dict, List

from numpy import array

from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.serde import parameters_to_proto
from flwr.proto.fleet_pb2 import (
    TokenizedTask,
)

flwr_parameters = ndarrays_to_parameters([array([1, 2, 3]), array([4, 5, 6])])
proto_parameters = parameters_to_proto(flwr_parameters)


def gen_basic_task(task_id: int = 42) -> TokenizedTask:
    tokenized_task = TokenizedTask()
    tokenized_task.token = uuid.uuid4().hex
    tokenized_task.task.task_id = task_id
    return tokenized_task


def gen_mock_messages() -> Dict[str, TokenizedTask]:
    mock_messages = {}
    # Reconnect
    mock_messages["reconnect"] = _gen_reconnect_tokenized_task()
    # Fit
    mock_messages["fit"] = _gen_fit_tokenized_task()
    return mock_messages


def _gen_reconnect_tokenized_task(reconnect_in: int = 5) -> TokenizedTask:
    _t = gen_basic_task()
    _t.task.legacy_server_message.reconnect_ins.seconds = reconnect_in
    return _t


def _gen_fit_tokenized_task() -> TokenizedTask:
    _t = gen_basic_task()
    _t.task.legacy_server_message.fit_ins.parameters.CopyFrom(proto_parameters)
    return _t
