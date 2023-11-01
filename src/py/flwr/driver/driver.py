# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
"""Flower driver service client."""


from threading import Lock
from typing import Iterable, List, Optional, Tuple, cast

from grpc import RpcError, StatusCode

from flwr.common.retry_invoker import RetryInvoker, exponential
from flwr.driver.grpc_driver import DEFAULT_SERVER_ADDRESS_DRIVER, GrpcDriver
from flwr.proto.driver_pb2 import (
    CreateWorkloadRequest,
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import TaskIns, TaskRes


class Driver:
    """`Driver` class provides an interface to the Driver API.

    Parameters
    ----------
    driver_service_address : Optional[str]
        The IPv4 or IPv6 address of the Driver API server.
        Defaults to `"[::]:9091"`.
    certificates : bytes (default: None)
        Tuple containing root certificate, server certificate, and private key
        to start a secure SSL-enabled server. The tuple is expected to have
        three bytes elements in the following order:

            * CA certificate.
            * server certificate.
            * server private key.
    invoker : Optional[RetryInvoker] (default: None)
        A `RetryInvoker` object to control the retry behavior on Driver API failures.
        If set to None, a default instance is created with an exponential backoff
        strategy, up to 10 attempts, a 300-second time limit, and retries are aborted
        only if the RpcError's status code is not `StatusCode.UNAVAILABLE`.
    """

    def __init__(
        self,
        driver_service_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        certificates: Optional[bytes] = None,
        invoker: Optional[RetryInvoker] = None,
    ) -> None:
        self.addr = driver_service_address
        self.certificates = certificates
        self.grpc_driver: Optional[GrpcDriver] = None
        self.workload_id: Optional[int] = None
        self.node = Node(node_id=0, anonymous=True)
        self.lock = Lock()
        # Initialize invoker
        if invoker is None:
            err_codes = (StatusCode.UNAVAILABLE,)
            invoker = RetryInvoker(
                exponential,
                RpcError,
                max_tries=10,
                max_time=300,
                should_giveup=lambda e: e.code() not in err_codes,  # type: ignore
            )
        self.invoker = invoker

    def _get_grpc_driver_and_workload_id(self) -> Tuple[GrpcDriver, int]:
        # Check if the GrpcDriver is initialized
        with self.lock:
            if self.grpc_driver is None or self.workload_id is None:
                # Connect and create workload
                self.grpc_driver = GrpcDriver(
                    driver_service_address=self.addr, certificates=self.certificates
                )
                self.grpc_driver.connect()
                res = self.grpc_driver.create_workload(CreateWorkloadRequest())
                self.workload_id = res.workload_id

        return self.grpc_driver, self.workload_id

    def get_nodes(self) -> List[Node]:
        """Get node IDs."""
        grpc_driver, workload_id = self._get_grpc_driver_and_workload_id()

        # Call GrpcDriver method
        req = GetNodesRequest(workload_id=workload_id)
        with self.lock:
            res = cast(
                GetNodesResponse, self.invoker.invoke(grpc_driver.get_nodes, req)
            )
        return list(res.nodes)

    def push_task_ins(self, task_ins_list: List[TaskIns]) -> List[str]:
        """Schedule tasks."""
        grpc_driver, workload_id = self._get_grpc_driver_and_workload_id()

        # Set workload_id
        for task_ins in task_ins_list:
            task_ins.workload_id = workload_id

        # Call GrpcDriver method
        req = PushTaskInsRequest(task_ins_list=task_ins_list)
        with self.lock:
            res = cast(
                PushTaskInsResponse, self.invoker.invoke(grpc_driver.push_task_ins, req)
            )
        return list(res.task_ids)

    def pull_task_res(self, task_ids: Iterable[str]) -> List[TaskRes]:
        """Get task results."""
        grpc_driver, _ = self._get_grpc_driver_and_workload_id()

        # Call GrpcDriver method
        req = PullTaskResRequest(node=self.node, task_ids=task_ids)
        with self.lock:
            res = cast(
                PullTaskResResponse, self.invoker.invoke(grpc_driver.pull_task_res, req)
            )
        return list(res.task_res_list)

    def __del__(self) -> None:
        """Disconnect GrpcDriver if connected."""
        # Check if GrpcDriver is initialized
        if self.grpc_driver is None:
            return

        # Disconnect
        with self.lock:
            self.grpc_driver.disconnect()
