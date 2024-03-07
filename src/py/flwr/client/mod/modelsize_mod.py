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
"""Model size mod."""


from logging import INFO

from flwr.client.typing import ClientAppCallable
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message


def modelsize_mod(msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
    """Model size mod.

    This mod logs the size of the model transmitted in the message.
    """
    if (
        msg.metadata.message_type == MessageType.TRAIN
        or msg.metadata.message_type == MessageType.EVALUATE
    ):

        model_size_stats = {}
        for record_name, parameters in msg.content.parameters_records.items():
            parameter_count = 0
            parameter_count_in_mb = 0
            for _, array in parameters.items():
                ndarray = array.numpy()
                parameter_count += ndarray.size
                parameter_count_in_mb += ndarray.size * ndarray.itemsize / 1024**2

            model_size_stats[f"{record_name}"] = {
                "parameters": parameter_count,
                "mb": parameter_count_in_mb,
            }

        log(INFO, model_size_stats)

    return call_next(msg, ctxt)
