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
"""Connection for a gRPC request-response channel to the SuperLink."""


import random
import threading
from contextlib import contextmanager
from copy import copy
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple, Type, Union, cast

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.client.heartbeat import start_ping_loop
from flwr.client.message_handler.message_handler import validate_out_message
from flwr.client.message_handler.task_handler import get_task_ins, validate_task_ins
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.constant import (
    PING_BASE_MULTIPLIER,
    PING_CALL_TIMEOUT,
    PING_DEFAULT_INTERVAL,
    PING_RANDOM_RANGE,
)
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.serde import (
    message_from_taskins,
    message_to_taskres,
    user_config_from_proto,
)
from flwr.common.typing import Fab, Run
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    DeleteNodeRequest,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PushTaskResRequest,
)
from flwr.proto.fleet_pb2_grpc import FleetStub  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .client_interceptor import AuthenticateClientInterceptor


class GrpcRereConnection(object):
    """Grpc-rere connection."""
    
    def __init__(  # pylint: disable=R0913, R0914, R0915
        server_address: str,
        insecure: bool,
        retry_invoker: RetryInvoker,
        max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
        root_certificates: Optional[Union[bytes, str]] = None,
        authentication_keys: Optional[
            Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
        ] = None,
    ) -> None:
        
        
    