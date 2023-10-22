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

from flwr.proto.task_pb2 import TaskIns, TaskRes


# pylint: disable-next=too-many-branches,too-many-statements
def validate_task_ins_or_res(tasks_ins_res: Union[TaskIns, TaskRes]) -> List[str]:
    """Validate a TaskIns or TaskRes."""
    validation_errors = []

    if tasks_ins_res.task_id != "":
        validation_errors.append("non-empty `task_id`")

    if not tasks_ins_res.HasField("task"):
        validation_errors.append("`task` does not set field `task`")

    # Created/delivered/TTL
    if tasks_ins_res.task.created_at != "":
        validation_errors.append("`created_at` must be an empty str")
    if tasks_ins_res.task.delivered_at != "":
        validation_errors.append("`delivered_at` must be an empty str")
    if tasks_ins_res.task.ttl != "":
        validation_errors.append("`ttl` must be an empty str")

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
        has_fields = {
            "sa": tasks_ins_res.task.HasField("sa"),
            "legacy_server_message": tasks_ins_res.task.HasField(
                "legacy_server_message"
            ),
        }
        if not (has_fields["sa"] or has_fields["legacy_server_message"]):
            err_msg = ", ".join([f"`{field}`" for field in has_fields])
            validation_errors.append(
                f"`task` in `TaskIns` must set at least one of fields {{{err_msg}}}"
            )
        if has_fields[
            "legacy_server_message"
        ] and not tasks_ins_res.task.legacy_server_message.HasField("msg"):
            validation_errors.append("`legacy_server_message` does not set field `msg`")

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
        has_fields = {
            "sa": tasks_ins_res.task.HasField("sa"),
            "legacy_client_message": tasks_ins_res.task.HasField(
                "legacy_client_message"
            ),
        }
        if not (has_fields["sa"] or has_fields["legacy_client_message"]):
            err_msg = ", ".join([f"`{field}`" for field in has_fields])
            validation_errors.append(
                f"`task` in `TaskRes` must set at least one of fields {{{err_msg}}}"
            )
        if has_fields[
            "legacy_client_message"
        ] and not tasks_ins_res.task.legacy_client_message.HasField("msg"):
            validation_errors.append("`legacy_client_message` does not set field `msg`")

        # Ancestors
        if len(tasks_ins_res.task.ancestry) == 0:
            validation_errors.append("`ancestry` is empty")

    return validation_errors
