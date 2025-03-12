# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower type definitions."""


from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]
NDArrays = list[NDArray]

# The following union type contains Python types corresponding to ProtoBuf types that
# ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
# not conform to other definitions of what a scalar is. Source:
# https://developers.google.com/protocol-buffers/docs/overview#scalar
Scalar = Union[bool, bytes, float, int, str]
Value = Union[
    bool,
    bytes,
    float,
    int,
    str,
    list[bool],
    list[bytes],
    list[float],
    list[int],
    list[str],
]

# Value types for common.MetricsRecord
MetricsScalar = Union[int, float]
MetricsScalarList = Union[list[int], list[float]]
MetricsRecordValues = Union[MetricsScalar, MetricsScalarList]
# Value types for common.ConfigsRecord
ConfigsScalar = Union[MetricsScalar, str, bytes, bool]
ConfigsScalarList = Union[MetricsScalarList, list[str], list[bytes], list[bool]]
ConfigsRecordValues = Union[ConfigsScalar, ConfigsScalarList]

Metrics = dict[str, Scalar]
MetricsAggregationFn = Callable[[list[tuple[int, Metrics]]], Metrics]

Config = dict[str, Scalar]
Properties = dict[str, Scalar]

# Value type for user configs
UserConfigValue = Union[bool, float, int, str]
UserConfig = dict[str, UserConfigValue]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


class ClientAppOutputCode(Enum):
    """ClientAppIO status codes."""

    SUCCESS = 0
    DEADLINE_EXCEEDED = 1
    UNKNOWN_ERROR = 2


@dataclass
class ClientAppOutputStatus:
    """ClientAppIO status."""

    code: ClientAppOutputCode
    message: str


@dataclass
class Parameters:
    """Model parameters."""

    tensors: list[bytes]
    tensor_type: str


@dataclass
class GetParametersIns:
    """Parameters request for a client."""

    config: Config


@dataclass
class GetParametersRes:
    """Response when asked to return parameters."""

    status: Status
    parameters: Parameters


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: dict[str, Scalar]


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: dict[str, Scalar]


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: dict[str, Scalar]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    status: Status
    loss: float
    num_examples: int
    metrics: dict[str, Scalar]


@dataclass
class GetPropertiesIns:
    """Properties request for a client."""

    config: Config


@dataclass
class GetPropertiesRes:
    """Properties response from a client."""

    status: Status
    properties: Properties


@dataclass
class ReconnectIns:
    """ReconnectIns message from server to client."""

    seconds: Optional[int]


@dataclass
class DisconnectRes:
    """DisconnectRes message from client to server."""

    reason: str


@dataclass
class ServerMessage:
    """ServerMessage is a container used to hold one instruction message."""

    get_properties_ins: Optional[GetPropertiesIns] = None
    get_parameters_ins: Optional[GetParametersIns] = None
    fit_ins: Optional[FitIns] = None
    evaluate_ins: Optional[EvaluateIns] = None


@dataclass
class ClientMessage:
    """ClientMessage is a container used to hold one result message."""

    get_properties_res: Optional[GetPropertiesRes] = None
    get_parameters_res: Optional[GetParametersRes] = None
    fit_res: Optional[FitRes] = None
    evaluate_res: Optional[EvaluateRes] = None


@dataclass
class RunStatus:
    """Run status information."""

    status: str
    sub_status: str
    details: str


@dataclass
class Run:  # pylint: disable=too-many-instance-attributes
    """Run details."""

    run_id: int
    fab_id: str
    fab_version: str
    fab_hash: str
    override_config: UserConfig
    pending_at: str
    starting_at: str
    running_at: str
    finished_at: str
    status: RunStatus

    @classmethod
    def create_empty(cls, run_id: int) -> "Run":
        """Return an empty Run instance."""
        return cls(
            run_id=run_id,
            fab_id="",
            fab_version="",
            fab_hash="",
            override_config={},
            pending_at="",
            starting_at="",
            running_at="",
            finished_at="",
            status=RunStatus(status="", sub_status="", details=""),
        )


@dataclass
class Fab:
    """Fab file representation."""

    hash_str: str
    content: bytes


class RunNotRunningException(BaseException):
    """Raised when a run is not running."""


class InvalidRunStatusException(BaseException):
    """Raised when an RPC is invalidated by the RunStatus."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# OIDC user authentication types
@dataclass
class UserAuthLoginDetails:
    """User authentication login details."""

    auth_type: str
    device_code: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass
class UserAuthCredentials:
    """User authentication tokens."""

    access_token: str
    refresh_token: str


@dataclass
class UserInfo:
    """User information for event log."""

    user_id: Optional[str]
    user_name: Optional[str]


@dataclass
class Actor:
    """Event log actor."""

    actor_id: Optional[str]
    description: Optional[str]
    ip_address: str


@dataclass
class Event:
    """Event log description."""

    action: str
    run_id: Optional[int]
    fab_hash: Optional[str]


@dataclass
class LogEntry:
    """Event log record."""

    timestamp: str
    actor: Actor
    event: Event
    status: str
