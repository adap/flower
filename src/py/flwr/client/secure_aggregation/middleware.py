from flwr.client.middleware import Layer, App
from flwr.client.workload_state import WorkloadState
from flwr.client.typing import Fwd, Bwd
from flwr.common.typing import FitRes, Value
from typing import Callable, Dict
from flwr.client.message_handler.task_handler import get_server_message_from_task_ins
from flwr.common import serde
from flwr.proto.task_pb2 import SecureAggregation, Task, TaskIns, TaskRes


def secure_aggregation_middleware(func: Callable[[WorkloadState, Dict[str, Value], Callable[[], FitRes]], Dict[str, Value]]):
    """."""
    def wrapper(fwd: Fwd, app: App) -> Bwd:
        # Ignore non-fit messages
        task_ins = fwd.task_ins
        server_msg = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)
        if server_msg is not None and server_msg.WhichOneof("msg") != "fit_ins":
            return app(fwd)
        
        named_values = serde.named_values_from_proto(task_ins.task.sa.named_values)
        state = fwd.state
        
        def fit() -> FitRes:
            # Raise an exception if TaskIns does not contain a FitIns
            if server_msg is None:
                raise ValueError("TaskIns does not contain a FitIns message.")
            fit_res_proto = app(fwd).task_res.task.legacy_client_message.fit_res
            return serde.fit_res_from_proto(fit_res_proto)
        
        res = func(state, named_values, fit)
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
            state=state,
        )
            