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
"""Middleware decorator for secure aggregation."""
from typing import Callable, Dict

from flwr.client.message_handler.task_handler import get_server_message_from_task_ins
from flwr.client.middleware import App, Layer
from flwr.client.typing import Bwd, Fwd
from flwr.client.workload_state import WorkloadState
from flwr.common import serde
from flwr.common.typing import FitRes, Value
from flwr.proto.task_pb2 import SecureAggregation, Task, TaskRes


def secure_aggregation_middleware(
    fn: Callable[
        [WorkloadState, Dict[str, Value], Callable[[], FitRes]], Dict[str, Value]
    ]
) -> Layer:
    """Wrap a function to create a secure aggregation middleware layer.

    This decorator transforms a user-defined function with a more intuitive
    signature into a middleware layer. The user-defined function should perform
    secure aggregation operations given the workload_state, named_values, and
    a fit function.

    Parameters
    ----------
    fn : Callable[
        [WorkloadState, Dict[str, Value], Callable[[], FitRes]], Dict[str, Value]
    ]
        The user-defined function to be transformed into a middleware layer.
        It takes three arguments:

            1. `WorkloadState`: The current workload state.
            2. `Dict[str, Value]`: A dictionary for aggregation from `sa` field in Task.
            3. `Callable[[], FitRes]`: A function that produces a FitRes when called.

        The function should return a `Dict[str, Value]`.

    Returns
    -------
    Layer
        A middleware layer.

    Example:
    ```
    @secure_aggregation_middleware
    def my_aggregation_logic(workload_state, named_values, fit):
        # Implement secure aggregation logic here
        return ret_named_values
    ```

    Note:
    The decorator is specifically designed for secure aggregation.
    It ignores non-fit messages.
    """

    def wrapper(fwd: Fwd, app: App) -> Bwd:
        # Ignore non-fit messages
        task_ins = fwd.task_ins
        server_msg = get_server_message_from_task_ins(
            task_ins, exclude_reconnect_ins=False
        )
        if server_msg is not None and server_msg.WhichOneof("msg") != "fit_ins":
            return app(fwd)

        named_values = serde.named_values_from_proto(task_ins.task.sa.named_values)
        workload_state = fwd.state

        # Create the fit function
        def fit() -> FitRes:
            # Raise an exception if TaskIns does not contain a FitIns
            if server_msg is None:
                raise ValueError("TaskIns does not contain a FitIns message.")
            fit_res_proto = app(fwd).task_res.task.legacy_client_message.fit_res
            return serde.fit_res_from_proto(fit_res_proto)

        res = fn(workload_state, named_values, fit)
        return Bwd(
            task_res=TaskRes(
                task_id="",
                group_id="",
                workload_id=0,
                task=Task(
                    ancestry=[],
                    sa=SecureAggregation(named_values=serde.named_values_to_proto(res)),
                ),
            ),
            state=workload_state,
        )

    return wrapper
