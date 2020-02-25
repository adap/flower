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
"""Provides class ClientProxy."""

from queue import Queue
from typing import Optional

from flower.proto.transport_pb2 import ClientRequest, ServerResponse


class ClientProxy:
    """ClientProxy holding requests and responses."""

    def __init__(self) -> None:
        """Create request/response queues."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self.requests: Queue[ClientRequest] = Queue(maxsize=1)
        self.responses: Queue[ServerResponse] = Queue(maxsize=1)

    def send_instruction_and_get_result(
        self, instruction: ServerResponse
    ) -> ClientRequest:
        """Return next request."""
        self.responses.put(instruction)
        return self.requests.get()

    def push_result_and_get_next_instruction(
        self, result: Optional[ClientRequest] = None
    ) -> ServerResponse:
        """Push result of last instruction and get next instruction of remote client.

        Args:
            result (ClientRequest, optional): Result of last instruction if available.
                This argument is optional as in case of the first remote client request
                no instruction was present and therefore no result should be pushed.

        Returns:
            ServerResponse: Next instruction to be processed by remote client
        """
        if result is not None:
            self.requests.put(result)

        return self.responses.get()
