from typing import List
import random
import time
from uuid import uuid4
from flwr.driver import Driver
from flwr.common import ServerMessage, FitIns, ndarrays_to_parameters, serde
from flwr.proto import driver_pb2, task_pb2, node_pb2, transport_pb2

from task import Net, get_parameters, set_parameters

# -------------------------------------------------------------------------- Driver SDK
driver = Driver(driver_service_address="[::]:9091", certificates=None)
# -------------------------------------------------------------------------- Driver SDK

sleep_time = 1
parameters = ndarrays_to_parameters(get_parameters(net=Net()))
num_rounds = 3
num_nodes = 1

# -------------------------------------------------------------------------- Driver SDK
driver.connect()
# -------------------------------------------------------------------------- Driver SDK

workload_id: str = str(uuid4())

for server_round in range(num_rounds):
    print(f"Commencing server round {server_round + 1}")

    # Schedule a task for three anonymous nodes
    fit_ins: FitIns = FitIns(parameters=parameters, config={})
    server_message = ServerMessage(fit_ins=fit_ins)
    server_message_proto: transport_pb2.ServerMessage = serde.server_message_to_proto(
        server_message=server_message
    )
    task_ins_list: List[task_pb2.TaskIns] = []
    for _ in range(num_nodes):
        new_task_ins = task_pb2.TaskIns(
            task_id="",  # Do not set, will be created and set by the DriverAPI
            group_id=str(server_round),
            workload_id=workload_id,
            task=task_pb2.Task(
                producer=node_pb2.Node(node_id=0, anonymous=True),
                consumer=node_pb2.Node(node_id=0, anonymous=True),
                legacy_server_message=server_message_proto,
            ),
        )
        task_ins_list.append(new_task_ins)

    push_task_ins_req = driver_pb2.PushTaskInsRequest(task_ins_list=task_ins_list)

    # ---------------------------------------------------------------------- Driver SDK
    push_task_ins_res: driver_pb2.PushTaskInsResponse = driver.push_task_ins(
        req=push_task_ins_req
    )
    # ---------------------------------------------------------------------- Driver SDK

    print(
        f"Scheduled {len(push_task_ins_res.task_ids)} tasks: {push_task_ins_res.task_ids}"
    )

    time.sleep(sleep_time)

    # Wait for results, ignore empty task_ids
    task_ids: List[str] = [
        task_id for task_id in push_task_ins_res.task_ids if task_id != ""
    ]
    all_task_res: List[task_pb2.TaskRes] = []
    while True:
        pull_task_res_req = driver_pb2.PullTaskResRequest(
            node=node_pb2.Node(node_id=0, anonymous=True),
            task_ids=task_ids,
        )

        # ------------------------------------------------------------------ Driver SDK
        pull_task_res_res: driver_pb2.PullTaskResResponse = driver.pull_task_res(
            req=pull_task_res_req
        )
        # ------------------------------------------------------------------ Driver SDK

        task_res_list: List[task_pb2.TaskRes] = pull_task_res_res.task_res_list
        print(f"Got {len(task_res_list)} results")

        time.sleep(sleep_time)

        all_task_res += task_res_list
        if len(all_task_res) == len(task_ids):
            break

    # "Aggregate" results
    node_messages = [task_res.task.legacy_client_message for task_res in all_task_res]
    print(f"Received {len(node_messages)} results")

    time.sleep(sleep_time)

    # Repeat

# -------------------------------------------------------------------------- Driver SDK
driver.disconnect()
# -------------------------------------------------------------------------- Driver SDK
