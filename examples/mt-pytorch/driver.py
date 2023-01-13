from typing import List
import random
import time

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

# -------------------------------------------------------------------------- Driver SDK
driver.connect()
# -------------------------------------------------------------------------- Driver SDK

for server_round in range(num_rounds):
    print(f"Commencing server round {server_round + 1}")

    # Get a list of node ID's from the server
    get_nodes_req = driver_pb2.GetNodesRequest()

    # ---------------------------------------------------------------------- Driver SDK
    get_nodes_res: driver_pb2.GetNodesResponse = driver.get_nodes(req=get_nodes_req)
    # ---------------------------------------------------------------------- Driver SDK

    # Sample three nodes
    all_node_ids: List[int] = get_nodes_res.node_ids
    print(f"Got {len(all_node_ids)} node IDs")
    num_node_ids_to_sample = 3 if len(all_node_ids) >= 3 else 1
    sampled_node_ids: List[int] = random.sample(all_node_ids, num_node_ids_to_sample)
    print(f"Sampled {len(sampled_node_ids)} node IDs: {sampled_node_ids}")

    time.sleep(sleep_time)

    # Schedule a task for all three nodes
    fit_ins: FitIns = FitIns(parameters=parameters, config={})
    server_message = ServerMessage(fit_ins=fit_ins)
    server_message_proto: transport_pb2.ServerMessage = serde.server_message_to_proto(
        server_message=server_message
    )
    task_ins_set: List[task_pb2.TaskIns] = []
    for sampled_node_id in sampled_node_ids:
        new_task_ins = task_pb2.TaskIns(
            task_id="",  # Will be created and set by the DriverAPI
            task=task_pb2.Task(
                producer=node_pb2.Node(node_id=0, anonymous=True),
                consumer=node_pb2.Node(node_id=sampled_node_id, anonymous=False),
                legacy_server_message=server_message_proto,
            ),
        )
        task_ins_set.append(new_task_ins)

    push_task_ins_req = driver_pb2.PushTaskInsRequest(task_ins_set=task_ins_set)

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
        pull_task_res_req = driver_pb2.PullTaskResRequest(task_ids=task_ids)

        # ------------------------------------------------------------------ Driver SDK
        pull_task_res_res: driver_pb2.PullTaskResResponse = driver.pull_task_res(
            req=pull_task_res_req
        )
        # ------------------------------------------------------------------ Driver SDK

        task_res_set: List[task_pb2.TaskRes] = pull_task_res_res.task_res_set
        print(f"Got {len(task_res_set)} results")

        time.sleep(sleep_time)

        all_task_res += task_res_set
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
