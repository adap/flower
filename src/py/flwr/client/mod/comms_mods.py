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
"""Model size and Message size mods."""

from logging import INFO
from sys import getsizeof

from flwr.client.typing import ClientAppCallable
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message


def message_size_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Message size mod.

    This mod logs the size in Bytes of the message being transmited.
    """
    message_size_in_bytes = 0


    # TODO:    
    
    log(INFO, "Message size: %i Bytes", message_size_in_bytes)

    return call_next(msg, ctxt)


def parameters_size_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Parameters size mod.

    This mod logs the number of parameters transmited in the message as well as their
    size in Bytes.
    """
    model_size_stats = {}
    for record_name, parameters in msg.content.parameters_records.items():
        parameter_count = 0
        parameter_count_in_bytes = 0
        for _, array in parameters.items():
            ndarray = array.numpy()
            parameter_count += ndarray.size
            parameter_count_in_bytes += ndarray.size * ndarray.itemsize

        model_size_stats[f"{record_name}"] = {
            "parameters": parameter_count,
            "bytes": parameter_count_in_bytes,
        }

    if model_size_stats:
        log(INFO, model_size_stats)

    return call_next(msg, ctxt)
