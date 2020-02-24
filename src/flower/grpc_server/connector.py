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
"""Provides class Connector handling request/response queuing and matching."""

from queue import Queue

from flower.proto.transport_pb2 import ClientRequest, ServerResponse


class Connector:
    """Connector handling request/response queuing and matching."""

    def __init__(self) -> None:
        """Create request/response queues."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self.requests: Queue[ClientRequest] = Queue(maxsize=1)
        self.responses: Queue[ServerResponse] = Queue(maxsize=1)

    def get_request(self, response: ServerResponse) -> ClientRequest:
        """Return next request."""
        self.responses.put(response)
        return self.requests.get()

    def get_response(self, request: ClientRequest) -> ServerResponse:
        """Return next response."""
        self.requests.put(request)
        return self.responses.get()
