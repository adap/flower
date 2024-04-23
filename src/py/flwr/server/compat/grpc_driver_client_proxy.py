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
"""Flower ClientProxy implementation for gRPC Driver API."""


import time
from typing import List, Optional

from flwr.common import DEFAULT_TTL, RecordSet, serde
from flwr.proto import driver_pb2, node_pb2, task_pb2  # pylint: disable=E0611

from ..driver.grpc_driver import GrpcDriverHelper
from .driver_client_proxy import DriverClientProxy, validate_task_res

SLEEP_TIME = 1


class GrpcDriverClientProxy(DriverClientProxy):
    """Flower client proxy which delegates work using the Driver API."""

    def __init__(
        self, node_id: int, driver: GrpcDriverHelper, anonymous: bool, run_id: int
    ):
        super().__init__(node_id=node_id, anonymous=anonymous, run_id=run_id)
        self.driver = driver

    def _send_receive_recordset(
        self,
        recordset: RecordSet,
        task_type: str,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> RecordSet:
        task_ins = task_pb2.TaskIns(  # pylint: disable=E1101
            task_id="",
            group_id=str(group_id) if group_id is not None else "",
            run_id=self.run_id,
            task=task_pb2.Task(  # pylint: disable=E1101
                producer=node_pb2.Node(  # pylint: disable=E1101
                    node_id=0,
                    anonymous=True,
                ),
                consumer=node_pb2.Node(  # pylint: disable=E1101
                    node_id=self.node_id,
                    anonymous=self.anonymous,
                ),
                task_type=task_type,
                recordset=serde.recordset_to_proto(recordset),
                ttl=DEFAULT_TTL,
            ),
        )

        # This would normally be recorded upon common.Message creation
        # but this compatibility stack doesn't create Messages,
        # so we need to inject `created_at` manually (needed for
        # taskins validation by server.utils.validator)
        task_ins.task.created_at = time.time()

        push_task_ins_req = driver_pb2.PushTaskInsRequest(  # pylint: disable=E1101
            task_ins_list=[task_ins]
        )

        # Send TaskIns to Driver API
        push_task_ins_res = self.driver.push_task_ins(req=push_task_ins_req)

        if len(push_task_ins_res.task_ids) != 1:
            raise ValueError("Unexpected number of task_ids")

        task_id = push_task_ins_res.task_ids[0]
        if task_id == "":
            raise ValueError(f"Failed to schedule task for node {self.node_id}")

        if timeout:
            start_time = time.time()

        while True:
            pull_task_res_req = driver_pb2.PullTaskResRequest(  # pylint: disable=E1101
                node=node_pb2.Node(node_id=0, anonymous=True),  # pylint: disable=E1101
                task_ids=[task_id],
            )

            # Ask Driver API for TaskRes
            pull_task_res_res = self.driver.pull_task_res(req=pull_task_res_req)

            task_res_list: List[task_pb2.TaskRes] = list(  # pylint: disable=E1101
                pull_task_res_res.task_res_list
            )
            if len(task_res_list) == 1:
                task_res = task_res_list[0]

                # This will raise an Exception if task_res carries an `error`
                validate_task_res(task_res=task_res)

                return serde.recordset_from_proto(task_res.task.recordset)

            if timeout is not None and time.time() > start_time + timeout:
                raise RuntimeError("Timeout reached")
            time.sleep(SLEEP_TIME)
