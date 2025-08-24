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

    for p_record in msg.content.parameters_records.values():
        message_size_in_bytes += p_record.count_bytes()

    for c_record in msg.content.configs_records.values():
        message_size_in_bytes += c_record.count_bytes()

    for m_record in msg.content.metrics_records.values():
        message_size_in_bytes += m_record.count_bytes()

    log(INFO, "Message size: %i bytes", message_size_in_bytes)

    return call_next(msg, ctxt)


def parameters_size_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Parameters size mod.

    This mod logs the number of parameters transmitted in the message as well as their
    size in bytes.
    """
    model_size_stats = {}
    parameters_size_in_bytes = 0
    for record_name, p_record in msg.content.parameters_records.items():
        p_record_bytes = p_record.count_bytes()
        parameters_size_in_bytes += p_record_bytes
        parameter_count = 0
        for array in p_record.values():
            parameter_count += (
                int(np.prod(array.shape)) if array.shape else array.numpy().size
            )

        model_size_stats[f"{record_name}"] = {
            "parameters": parameter_count,
            "bytes": p_record_bytes,
        }

    if model_size_stats:
        log(INFO, model_size_stats)

    log(INFO, "Total parameters transmitted: %i bytes", parameters_size_in_bytes)

    return call_next(msg, ctxt)
