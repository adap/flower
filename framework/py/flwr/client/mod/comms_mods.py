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
"""Mods that report statistics about message communication."""


from logging import INFO

import numpy as np

from flwr.client.typing import ClientAppCallable
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message


def message_size_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Message size mod.

    This mod logs the size in bytes of the message being transmited.
    """
    message_size_in_bytes = 0

    for record in msg.content.values():
        message_size_in_bytes += record.count_bytes()

    log(INFO, "Message size: %i bytes", message_size_in_bytes)

    return call_next(msg, ctxt)


def arrays_size_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Arrays size mod.

    This mod logs the number of array elements transmitted in ``ArrayRecord`` objects
    of the message as well as their sizes in bytes.
    """
    model_size_stats = {}
    arrays_size_in_bytes = 0
    for record_name, arr_record in msg.content.array_records.items():
        arr_record_bytes = arr_record.count_bytes()
        arrays_size_in_bytes += arr_record_bytes
        element_count = 0
        for array in arr_record.values():
            element_count += (
                int(np.prod(array.shape)) if array.shape else array.numpy().size
            )

        model_size_stats[f"{record_name}"] = {
            "elements": element_count,
            "bytes": arr_record_bytes,
        }

    if model_size_stats:
        log(INFO, model_size_stats)

    log(INFO, "Total array elements transmitted: %i bytes", arrays_size_in_bytes)

    return call_next(msg, ctxt)
