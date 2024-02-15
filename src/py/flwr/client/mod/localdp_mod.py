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
"""Local DP modifier."""


from flwr.client.typing import ClientAppCallable
from flwr.common.context import Context
from flwr.common.message import Message


class localdp_mod:
    def __init__(self, clipping_norm: float, ):


    def __call__(self, msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
        if msg.metadata.task_type == TASK_TYPE_FIT:
            fit_ins = compat.recordset_to_fitins(msg.message, keep_input=True)
