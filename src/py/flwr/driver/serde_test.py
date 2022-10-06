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
"""Tests for driver serialization."""


from flwr.driver import serde
from flwr.driver.messages import GetClientsRequest, GetClientsResponse


def test_get_clients_request_serde() -> None:
    """Test status message (de-)serialization."""

    # Prepare
    req = GetClientsRequest()

    # Execute
    msg = serde.get_clients_request_to_proto(req)
    req_actual = serde.get_clients_request_from_proto(msg)

    # Assert
    assert req_actual == req


def test_get_clients_response_serde() -> None:
    """Test status message (de-)serialization."""

    # Prepare
    res = GetClientsResponse(client_ids=[1, 2, 3])

    # Execute
    msg = serde.get_clients_response_to_proto(res)
    res_actual = serde.get_clients_response_from_proto(msg)

    # Assert
    assert res_actual == res
