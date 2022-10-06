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
"""Flower driver serialization."""


from flwr.driver.messages import GetClientsRequest, GetClientsResponse
from flwr.proto import driver_pb2


def get_clients_request_to_proto(
    req: GetClientsRequest,
) -> driver_pb2.GetClientsRequest:
    """."""
    # pylint: disable=W0613
    return driver_pb2.GetClientsRequest()


def get_clients_request_from_proto(
    msg: driver_pb2.GetClientsRequest,
) -> GetClientsRequest:
    """."""
    # pylint: disable=W0613
    return GetClientsRequest()


def get_clients_response_to_proto(
    res: GetClientsResponse,
) -> driver_pb2.GetClientsResponse:
    """."""
    return driver_pb2.GetClientsResponse(client_ids=res.client_ids)


def get_clients_response_from_proto(
    msg: driver_pb2.GetClientsResponse,
) -> GetClientsResponse:
    """."""
    return GetClientsResponse(client_ids=list(msg.client_ids))
