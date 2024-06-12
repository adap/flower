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
"""Flower command line interface `log` command."""

import time
from logging import DEBUG, INFO

import grpc
import typer
from typing_extensions import Annotated

from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log as logger

STREAM_DURATION: int = 60


def log(
    run_id: Annotated[
        int,
        typer.Option(case_sensitive=False, help="The Flower run ID to query"),
    ],
    follow: Annotated[
        bool,
        typer.Option(case_sensitive=False, help="Use this flag to follow logstream"),
    ] = True,
) -> None:
    """Get logs from Flower run."""

    def on_channel_state_change(channel_connectivity: str) -> None:
        """Log channel connectivity."""
        logger(DEBUG, channel_connectivity)

    # pylint: disable=unused-argument
    def stream_logs(run_id: int, channel: grpc.Channel, duration: int) -> None:
        """Stream logs from the beginning of a run with connection refresh."""

    # pylint: disable=unused-argument
    def print_logs(run_id: int, channel: grpc.Channel, timeout: int) -> None:
        """Print logs from the beginning of a run."""

    channel = create_channel(
        server_address="127.0.0.1:9093",
        insecure=True,
        root_certificates=None,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)

    if follow:
        try:
            while True:
                logger(INFO, "Streaming logs")
                stream_logs(run_id, channel, STREAM_DURATION)
                time.sleep(2)
                logger(INFO, "Reconnecting to logstream")
        except KeyboardInterrupt:
            logger(INFO, "Exiting logstream")
            channel.close()
    else:
        print_logs(run_id, channel, timeout=1)
