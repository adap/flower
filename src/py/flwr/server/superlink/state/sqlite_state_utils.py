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
"""Utility functions for SQLite based implemenation of server state."""


import sqlite3
from typing import Any

from flwr.common.constant import Status
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.recordset_pb2 import RecordSet  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes  # pylint: disable=E0611

from .utils import convert_sint64_to_uint64


def dict_factory(
    cursor: sqlite3.Cursor,
    row: sqlite3.Row,
) -> dict[str, Any]:
    """Turn SQLite results into dicts.

    Less efficent for retrival of large amounts of data but easier to use.
    """
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row))


def task_ins_to_dict(task_msg: TaskIns) -> dict[str, Any]:
    """Transform TaskIns to dict."""
    result = {
        "task_id": task_msg.task_id,
        "group_id": task_msg.group_id,
        "run_id": task_msg.run_id,
        "producer_anonymous": task_msg.task.producer.anonymous,
        "producer_node_id": task_msg.task.producer.node_id,
        "consumer_anonymous": task_msg.task.consumer.anonymous,
        "consumer_node_id": task_msg.task.consumer.node_id,
        "created_at": task_msg.task.created_at,
        "delivered_at": task_msg.task.delivered_at,
        "pushed_at": task_msg.task.pushed_at,
        "ttl": task_msg.task.ttl,
        "ancestry": ",".join(task_msg.task.ancestry),
        "task_type": task_msg.task.task_type,
        "recordset": task_msg.task.recordset.SerializeToString(),
    }
    return result


def task_res_to_dict(task_msg: TaskRes) -> dict[str, Any]:
    """Transform TaskRes to dict."""
    result = {
        "task_id": task_msg.task_id,
        "group_id": task_msg.group_id,
        "run_id": task_msg.run_id,
        "producer_anonymous": task_msg.task.producer.anonymous,
        "producer_node_id": task_msg.task.producer.node_id,
        "consumer_anonymous": task_msg.task.consumer.anonymous,
        "consumer_node_id": task_msg.task.consumer.node_id,
        "created_at": task_msg.task.created_at,
        "delivered_at": task_msg.task.delivered_at,
        "pushed_at": task_msg.task.pushed_at,
        "ttl": task_msg.task.ttl,
        "ancestry": ",".join(task_msg.task.ancestry),
        "task_type": task_msg.task.task_type,
        "recordset": task_msg.task.recordset.SerializeToString(),
    }
    return result


def dict_to_task_ins(task_dict: dict[str, Any]) -> TaskIns:
    """Turn task_dict into protobuf message."""
    recordset = RecordSet()
    recordset.ParseFromString(task_dict["recordset"])

    result = TaskIns(
        task_id=task_dict["task_id"],
        group_id=task_dict["group_id"],
        run_id=task_dict["run_id"],
        task=Task(
            producer=Node(
                node_id=task_dict["producer_node_id"],
                anonymous=task_dict["producer_anonymous"],
            ),
            consumer=Node(
                node_id=task_dict["consumer_node_id"],
                anonymous=task_dict["consumer_anonymous"],
            ),
            created_at=task_dict["created_at"],
            delivered_at=task_dict["delivered_at"],
            pushed_at=task_dict["pushed_at"],
            ttl=task_dict["ttl"],
            ancestry=task_dict["ancestry"].split(","),
            task_type=task_dict["task_type"],
            recordset=recordset,
        ),
    )
    return result


def dict_to_task_res(task_dict: dict[str, Any]) -> TaskRes:
    """Turn task_dict into protobuf message."""
    recordset = RecordSet()
    recordset.ParseFromString(task_dict["recordset"])

    result = TaskRes(
        task_id=task_dict["task_id"],
        group_id=task_dict["group_id"],
        run_id=task_dict["run_id"],
        task=Task(
            producer=Node(
                node_id=task_dict["producer_node_id"],
                anonymous=task_dict["producer_anonymous"],
            ),
            consumer=Node(
                node_id=task_dict["consumer_node_id"],
                anonymous=task_dict["consumer_anonymous"],
            ),
            created_at=task_dict["created_at"],
            delivered_at=task_dict["delivered_at"],
            pushed_at=task_dict["pushed_at"],
            ttl=task_dict["ttl"],
            ancestry=task_dict["ancestry"].split(","),
            task_type=task_dict["task_type"],
            recordset=recordset,
        ),
    )
    return result


def determine_run_status(row: dict[str, Any]) -> str:
    """Determine the status of the run based on timestamp fields."""
    if row["starting_at"]:
        if row["running_at"]:
            if row["finished_at"]:
                return Status.FINISHED
            return Status.RUNNING
        return Status.STARTING
    run_id = convert_sint64_to_uint64(row["run_id"])
    raise sqlite3.IntegrityError(f"The run {run_id} does not have a valid status.")
