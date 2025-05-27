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
"""ClientApp Scheduler."""


from flwr.common.constant import CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS
from typing import Optional, Union
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
import multiprocessing
import threading
import time
import os
from flwr.client.clientapp.app import run_clientapp
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.clientappio_pb2 import (
    GetRunIdsWithPendingMessagesRequest, 
    GetRunIdsWithPendingMessagesResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)


def run_scheduler(
    supernode_clientappio_api_address: str = CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
):
    """Run the ClientApp scheduler."""
    print("Scheduler started")
    # Initialize gRPC stub
    channel = create_channel(
        supernode_clientappio_api_address,
        insecure=True,
    )
    channel.subscribe(on_channel_state_change)
    stub = ClientAppIoStub(channel)
    
    # Initialize multiprocessing context
    mp_ctx = multiprocessing.get_context("spawn")

    try:
        while True:
            res: GetRunIdsWithPendingMessagesResponse = stub.GetRunIdsWithPendingMessages(
                GetRunIdsWithPendingMessagesRequest()
            )
            print(f"Scheduler: Found runs: {res.run_ids}")

            # Get the first run ID available
            for run_id in res.run_ids:
                print(f"Scheduler: Processing run ID {run_id}")
                tk_req = RequestTokenRequest(run_id=run_id)
                print(f"Scheduler: Requesting token for run ID {run_id}")
                tk_res: RequestTokenResponse = stub.RequestToken(tk_req)
                print(f"Scheduler: Received token for run ID {run_id}: {tk_res.token}")
                proc = mp_ctx.Process(
                    target=_run_clientapp_with_monitoring,
                    args=(
                        os.getpid(),
                        supernode_clientappio_api_address,
                        run_id,
                        tk_res.token,
                    ),
                    daemon=True,
                )
                proc.start()
                proc.join()

            time.sleep(1)
    finally:
        channel.close()


def _run_clientapp_with_monitoring(
    main_pid: int,
    clientappio_api_address: str,
    run_id: int,
    token: str,
):
    """Run the ClientApp with monitoring."""
    def main_process_monitor() -> None:
        while True:
            time.sleep(1)
            if os.getppid() != main_pid:
                os.kill(os.getpid(), 9)

    threading.Thread(target=main_process_monitor, daemon=True).start()

    run_clientapp(
        clientappio_api_address=clientappio_api_address,
        run_id=run_id,
        token=token,
    )

