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
"""Abstract class for Flower Event Log Writer Plugin."""


from abc import ABC, abstractmethod
from typing import Union

import grpc

from flwr.common.typing import LogEntry, UserInfo
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    ListRunsRequest,
    ListRunsResponse,
    StartRunRequest,
    StartRunResponse,
    StopRunRequest,
    StopRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)

# Type variables for request and response messages
EventLogRequest = Union[
    StartRunRequest,
    ListRunsRequest,
    StreamLogsRequest,
    StopRunRequest,
    GetLoginDetailsRequest,
]
EventLogResponse = Union[
    StartRunResponse,
    ListRunsResponse,
    StreamLogsResponse,
    StopRunResponse,
    GetLoginDetailsResponse,
]


class EventLogWriterPlugin(ABC):
    """Abstract Flower Event Log Writer Plugin class for ExecServicer."""

    @abstractmethod
    def __init__(self) -> None:
        """Abstract constructor."""

    @abstractmethod
    def compose_log_before_event(  # pylint: disable=too-many-arguments
        self,
        request: EventLogRequest,
        context: grpc.ServicerContext,
        user_info: UserInfo,
        method_name: str,
    ) -> LogEntry:
        """Compose pre-event log entry from the provided request and context."""

    @abstractmethod
    def compose_log_after_event(  # pylint: disable=too-many-arguments,R0917
        self,
        request: EventLogRequest,
        context: grpc.ServicerContext,
        user_info: UserInfo,
        method_name: str,
        response: EventLogResponse,
    ) -> LogEntry:
        """Compose post-event log entry from the provided response and context."""

    @abstractmethod
    def write_log(
        self,
        log_entry: LogEntry,
    ) -> None:
        """Write the event log to the specified data sink."""
