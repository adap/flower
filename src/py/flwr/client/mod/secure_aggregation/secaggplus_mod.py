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
"""Modifier for the SecAgg+ protocol."""


from logging import WARNING
from typing import cast

from flwr.client.typing import ClientAppCallable
from flwr.common import Context, Message, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import recordset_compat as compat
from flwr.common.constant import MessageType
from flwr.common.logger import log
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    parameters_multiply,
)
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key

from .secaggplus_base_mod import secaggplus_base_mod


def secaggplus_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Handle incoming message and return results, following the SecAgg+ protocol.

    This modifier extends `secaggplus_base_mod` by calculating the weighted average.
    It should be used with the `SecAggPlusWorkflow` on the ServerApp side.
    """
    # Ignore non-fit messages
    if msg.metadata.message_type != MessageType.TRAIN:
        return call_next(msg, ctxt)

    def new_call_next(_msg: Message, _ctxt: Context) -> Message:
        # Retrieve configs
        cfg = _msg.content.configs_records[RECORD_KEY_CONFIGS]
        clipping_range = cast(float, cfg[Key.CLIPPING_RANGE])
        max_weight = cast(float, cfg[Key.MAX_WEIGHT])

        # Modify the reply message
        out_msg = call_next(_msg, _ctxt)
        out_content = out_msg.content
        fitres = compat.recordset_to_fitres(out_content, keep_input=True)

        # Scale the parameters
        ratio = fitres.num_examples / max_weight
        scaled_ratio = ratio * clipping_range
        if ratio > 1:
            log(
                WARNING,
                "Potential overflow warning: the provided weight (%s) exceeds the "
                "specified max_weight (%s). This may lead to overflow issues.",
                fitres.num_examples,
                max_weight,
            )
        weights = parameters_to_ndarrays(fitres.parameters)
        weights = parameters_multiply(weights, ratio)
        weights = factor_combine(scaled_ratio, weights)
        fitres.parameters = ndarrays_to_parameters(weights)
        out_msg.content = compat.fitres_to_recordset(fitres, keep_input=True)
        return out_msg

    return secaggplus_base_mod(msg, ctxt, new_call_next)
