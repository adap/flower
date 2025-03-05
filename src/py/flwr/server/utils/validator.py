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


import time
from typing import Union

from flwr.common import Message
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611


# pylint: disable-next=too-many-branches,too-many-statements
def validate_task_ins_or_res(tasks_ins_res: Union[TaskIns, TaskRes]) -> list[str]:
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

    # Verify TTL and created_at time
    current_time = time.time()
    if tasks_ins_res.task.created_at + tasks_ins_res.task.ttl <= current_time:
        validation_errors.append("Task TTL has expired")

    # TaskIns specific
    if isinstance(tasks_ins_res, TaskIns):
        # Task producer
        if not tasks_ins_res.task.HasField("producer"):
            validation_errors.append("`producer` does not set field `producer`")
        if tasks_ins_res.task.producer.node_id != SUPERLINK_NODE_ID:
            validation_errors.append(f"`producer.node_id` is not {SUPERLINK_NODE_ID}")

        # Task consumer
        if not tasks_ins_res.task.HasField("consumer"):
            validation_errors.append("`consumer` does not set field `consumer`")
        if tasks_ins_res.task.consumer.node_id == SUPERLINK_NODE_ID:
            validation_errors.append("consumer MUST provide a valid `node_id`")

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
        if tasks_ins_res.task.producer.node_id == SUPERLINK_NODE_ID:
            validation_errors.append("producer MUST provide a valid `node_id`")

        # Task consumer
        if not tasks_ins_res.task.HasField("consumer"):
            validation_errors.append("`consumer` does not set field `consumer`")
        if tasks_ins_res.task.consumer.node_id != SUPERLINK_NODE_ID:
            validation_errors.append(f"consumer is not {SUPERLINK_NODE_ID}")

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


# pylint: disable-next=too-many-branches
def validate_message(message: Message, is_reply_message: bool) -> list[str]:
    """Validate a Message."""
    validation_errors = []
    metadata = message.metadata

    if metadata.message_id != "":
        validation_errors.append("non-empty `metadata.message_id`")

    # Created/delivered/TTL/Pushed
    if (
        metadata.created_at < 1740700800.0
    ):  # unix timestamp of 28 February 2025 00h:00m:00s UTC
        validation_errors.append(
            "`metadata.created_at` must be a float that records the unix timestamp "
            "in seconds when the message was created."
        )
    if metadata.delivered_at != "":
        validation_errors.append("`metadata.delivered_at` must be an empty str")
    if metadata.ttl <= 0:
        validation_errors.append("`metadata.ttl` must be higher than zero")

    # Verify TTL and created_at time
    current_time = time.time()
    if metadata.created_at + metadata.ttl <= current_time:
        validation_errors.append("Message TTL has expired")

    # Source node is set and is not zero
    if not metadata.src_node_id:
        validation_errors.append("`metadata.src_node_id` is not set.")

    # Destination node is set and is not zero
    if not metadata.dst_node_id:
        validation_errors.append("`metadata.dst_node_id` is not set.")

    # Message type
    if metadata.message_type == "":
        validation_errors.append("`metadata.message_type` MUST be set")

    # Content
    if not message.has_content() != message.has_error():
        validation_errors.append(
            "Either message `content` or `error` MUST be set (but not both)"
        )

    # Link respose to original message
    if not is_reply_message:
        if metadata.reply_to_message != "":
            validation_errors.append("`metadata.reply_to_message` MUST not be set.")
        if metadata.src_node_id != SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.src_node_id` is not {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )
        if metadata.dst_node_id == SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.dst_node_id` is {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )
    else:
        if metadata.reply_to_message == "":
            validation_errors.append("`metadata.reply_to_message` MUST be set.")
        if metadata.src_node_id == SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.src_node_id` is {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )
        if metadata.dst_node_id != SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.dst_node_id` is not {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )

    return validation_errors
