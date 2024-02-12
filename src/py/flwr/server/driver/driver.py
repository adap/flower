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


from typing import Iterable, List, Optional, Tuple

from flwr.proto.driver_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    GetNodesRequest,
    PullTaskResRequest,
    PushTaskInsRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611
from flwr.server.driver.grpc_driver import DEFAULT_SERVER_ADDRESS_DRIVER, GrpcDriver


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
    """

    def __init__(
        self,
        driver_service_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        root_certificates: Optional[bytes] = None,
    ) -> None:
        self.addr = driver_service_address
        self.root_certificates = root_certificates
        self.grpc_driver: Optional[GrpcDriver] = None
        self.run_id: Optional[int] = None
        self.node = Node(node_id=0, anonymous=True)

    def _get_grpc_driver_and_run_id(self) -> Tuple[GrpcDriver, int]:
        # Check if the GrpcDriver is initialized
        if self.grpc_driver is None or self.run_id is None:
            # Connect and create run
            self.grpc_driver = GrpcDriver(
                driver_service_address=self.addr,
                root_certificates=self.root_certificates,
            )
            self.grpc_driver.connect()
            res = self.grpc_driver.create_run(CreateRunRequest())
            self.run_id = res.run_id

        return self.grpc_driver, self.run_id

    def get_nodes(self) -> List[Node]:
        """Get node IDs."""
        grpc_driver, run_id = self._get_grpc_driver_and_run_id()

        # Call GrpcDriver method
        res = grpc_driver.get_nodes(GetNodesRequest(run_id=run_id))
        return list(res.nodes)

    def push_task_ins(self, task_ins_list: List[TaskIns]) -> List[str]:
        """Schedule tasks."""
        grpc_driver, run_id = self._get_grpc_driver_and_run_id()

        # Set run_id
        for task_ins in task_ins_list:
            task_ins.run_id = run_id

        # Call GrpcDriver method
        res = grpc_driver.push_task_ins(PushTaskInsRequest(task_ins_list=task_ins_list))
        return list(res.task_ids)

    def pull_task_res(self, task_ids: Iterable[str]) -> List[TaskRes]:
        """Get task results."""
        grpc_driver, _ = self._get_grpc_driver_and_run_id()

        # Call GrpcDriver method
        res = grpc_driver.pull_task_res(
            PullTaskResRequest(node=self.node, task_ids=task_ids)
        )
        return list(res.task_res_list)

    def __del__(self) -> None:
        """Disconnect GrpcDriver if connected."""
        # Check if GrpcDriver is initialized
        if self.grpc_driver is None:
            return

        # Disconnect
        self.grpc_driver.disconnect()
