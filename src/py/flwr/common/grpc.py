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
"""gRPC-related utility functions."""


from logging import INFO
from typing import Optional

import grpc

from flwr.common.logger import log

GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # == 512 * 1024 * 1024


def create_channel(
    server_address: str,
    root_certificates: Optional[bytes] = None,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
) -> grpc.Channel:
    """Create a gRPC channel, either secure or insecure."""

    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    channel_options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    if root_certificates is not None:
        ssl_channel_credentials = grpc.ssl_channel_credentials(root_certificates)
        channel = grpc.secure_channel(
            server_address, ssl_channel_credentials, options=channel_options
        )
        log(INFO, "Opened secure gRPC connection using certificates")
    else:
        channel = grpc.insecure_channel(server_address, options=channel_options)
        log(INFO, "Opened insecure gRPC connection (no certificates were passed)")

    return channel
