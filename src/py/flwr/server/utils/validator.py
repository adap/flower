# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Validators."""


from typing import List, Union

from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611


# pylint: disable-next=too-many-branches,too-many-statements
def validate_task_ins_or_res(tasks_ins_res: Union[TaskIns, TaskRes]) -> List[str]:
    """Validate a TaskIns or TaskRes."""
    validation_errors = []

    if tasks_ins_res.task_id != "":
        validation_errors.append("non-empty `task_id`")

    if not tasks_ins_res.HasField("task"):
        validation_errors.append("`task` does not set field `task`")

    # Created/delivered/TTL/Pushed
    if (
        tasks_ins_res.task.created_at < 1711497600.0
    ):  # unix timestamp of 27 March 2024 00h:00m:00s UTC
        validation_errors.append(
            "`created_at` must be a float that records the unix timestamp "
            "in seconds when the message was created."
        )
    if tasks_ins_res.task.delivered_at != "":
        validation_errors.append("`delivered_at` must be an empty str")
    if tasks_ins_res.task.ttl <= 0:
        validation_errors.append("`ttl` must be higher than zero")
    if tasks_ins_res.task.pushed_at < 1711497600.0:
        # unix timestamp of 27 March 2024 00h:00m:00s UTC
        validation_errors.append("`pushed_at` is not a recent timestamp")

    # TaskIns specific
    if isinstance(tasks_ins_res, TaskIns):
        # Task producer
        if not tasks_ins_res.task.HasField("producer"):
            validation_errors.append("`producer` does not set field `producer`")
        if tasks_ins_res.task.producer.node_id != 0:
            validation_errors.append("`producer.node_id` is not 0")
        if not tasks_ins_res.task.producer.anonymous:
            validation_errors.append("`producer` is not anonymous")

        # Task consumer
        if not tasks_ins_res.task.HasField("consumer"):
            validation_errors.append("`consumer` does not set field `consumer`")
        if (
            tasks_ins_res.task.consumer.anonymous
            and tasks_ins_res.task.consumer.node_id != 0
        ):
            validation_errors.append("anonymous consumers MUST NOT set a `node_id`")
        if (
            not tasks_ins_res.task.consumer.anonymous
            and tasks_ins_res.task.consumer.node_id == 0
        ):
            validation_errors.append("non-anonymous consumer MUST provide a `node_id`")

        # Content check
        if tasks_ins_res.task.task_type == "":
            validation_errors.append("`task_type` MUST be set")
        if not (
            tasks_ins_res.task.HasField("recordset")
            ^ tasks_ins_res.task.HasField("error")
        ):
            validation_errors.append("Either `recordset` or `error` MUST be set")

        # Ancestors
        if len(tasks_ins_res.task.ancestry) != 0:
            validation_errors.append("`ancestry` is not empty")

    # TaskRes specific
    if isinstance(tasks_ins_res, TaskRes):
        # Task producer
        if not tasks_ins_res.task.HasField("producer"):
            validation_errors.append("`producer` does not set field `producer`")
        if (
            tasks_ins_res.task.producer.anonymous
            and tasks_ins_res.task.producer.node_id != 0
        ):
            validation_errors.append("anonymous producers MUST NOT set a `node_id`")
        if (
            not tasks_ins_res.task.producer.anonymous
            and tasks_ins_res.task.producer.node_id == 0
        ):
            validation_errors.append("non-anonymous producer MUST provide a `node_id`")

        # Task consumer
        if not tasks_ins_res.task.HasField("consumer"):
            validation_errors.append("`consumer` does not set field `consumer`")
        if (
            tasks_ins_res.task.consumer.anonymous
            and tasks_ins_res.task.consumer.node_id != 0
        ):
            validation_errors.append("anonymous consumers MUST NOT set a `node_id`")
        if (
            not tasks_ins_res.task.consumer.anonymous
            and tasks_ins_res.task.consumer.node_id == 0
        ):
            validation_errors.append("non-anonymous consumer MUST provide a `node_id`")

        # Content check
        if tasks_ins_res.task.task_type == "":
            validation_errors.append("`task_type` MUST be set")
        if not (
            tasks_ins_res.task.HasField("recordset")
            ^ tasks_ins_res.task.HasField("error")
        ):
            validation_errors.append("Either `recordset` or `error` MUST be set")

        # Ancestors
        if len(tasks_ins_res.task.ancestry) == 0:
            validation_errors.append("`ancestry` is empty")

    return validation_errors
