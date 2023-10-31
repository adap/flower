import random
import time
from typing import Dict, List, Tuple

import numpy as np
from workflows import get_workflow_factory

from flwr.common import Metrics, ndarrays_to_parameters
from flwr.driver import GrpcDriver
from flwr.proto import driver_pb2, node_pb2, task_pb2
from flwr.server import History


# Convert instruction/result dict to/from list of TaskIns/TaskRes
def task_dict_to_task_ins_list(
    task_dict: Dict[int, task_pb2.Task]
) -> List[task_pb2.TaskIns]:
    def merge(_task: task_pb2.Task, _merge_task: task_pb2.Task) -> task_pb2.Task:
        _task.MergeFrom(_merge_task)
        return _task

    return [
        task_pb2.TaskIns(
            task_id="",  # Do not set, will be created and set by the DriverAPI
            group_id="",
            workload_id=workload_id,
            task=merge(
                task,
                task_pb2.Task(
                    producer=node_pb2.Node(
                        node_id=0,
                        anonymous=True,
                    ),
                    consumer=node_pb2.Node(
                        node_id=sampled_node_id,
                        # Must be False for this Secure Aggregation example
                        anonymous=False,
                    ),
                ),
            ),
        )
        for sampled_node_id, task in task_dict.items()
    ]


def task_res_list_to_task_dict(
    task_res_list: List[task_pb2.TaskRes],
) -> Dict[int, task_pb2.Task]:
    return {task_res.task.producer.node_id: task_res.task for task_res in task_res_list}


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


# -------------------------------------------------------------------------- Driver SDK
driver = GrpcDriver(driver_service_address="0.0.0.0:9091", certificates=None)
# -------------------------------------------------------------------------- Driver SDK

anonymous_client_nodes = False
num_client_nodes_per_round = 5
sleep_time = 0.5
time_out = 3.9
num_rounds = 3
parameters = ndarrays_to_parameters([np.ones(3)])
wf_factory = get_workflow_factory()

# -------------------------------------------------------------------------- Driver SDK
driver.connect()
create_workload_res: driver_pb2.CreateWorkloadResponse = driver.create_workload(
    req=driver_pb2.CreateWorkloadRequest()
)
# -------------------------------------------------------------------------- Driver SDK

workload_id = create_workload_res.workload_id
print(f"Created workload id {workload_id}")

history = History()
for server_round in range(num_rounds):
    print(f"Commencing server round {server_round + 1}")

    # List of sampled node IDs in this round
    sampled_node_ids: List[int] = []

    # Sample node ids
    if anonymous_client_nodes:
        # If we're working with anonymous clients, we don't know their identities, and
        # we don't know how many of them we have. We, therefore, have to assume that
        # enough anonymous client nodes are available or become available over time.
        #
        # To schedule a TaskIns for an anonymous client node, we set the node_id to 0
        # (and `anonymous` to True)
        # Here, we create an array with only zeros in it:
        sampled_node_ids = [0] * num_client_nodes_per_round
    else:
        # If our client nodes have identiy (i.e., they are not anonymous), we can get
        # those IDs from the Driver API using `get_nodes`. If enough clients are
        # available via the Driver API, we can select a subset by taking a random
        # sample.
        #
        # The Driver API might not immediately return enough client node IDs, so we
        # loop and wait until enough client nodes are available.
        while True:
            # Get a list of node ID's from the server
            get_nodes_req = driver_pb2.GetNodesRequest(workload_id=workload_id)

            # ---------------------------------------------------------------------- Driver SDK
            get_nodes_res: driver_pb2.GetNodesResponse = driver.get_nodes(
                req=get_nodes_req
            )
            # ---------------------------------------------------------------------- Driver SDK

            all_node_ids: List[int] = [node.node_id for node in get_nodes_res.nodes]

            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                sampled_node_ids = random.sample(
                    all_node_ids, num_client_nodes_per_round
                )
                break

            time.sleep(3)

    # Log sampled node IDs
    time.sleep(sleep_time)

    workflow = wf_factory(parameters, sampled_node_ids)
    node_messages = None

    while True:
        try:
            instructions: Dict[int, task_pb2.Task] = workflow.send(node_messages)
            next(workflow)
        except StopIteration:
            break
        # Schedule a task for all sampled nodes
        task_ins_list: List[task_pb2.TaskIns] = task_dict_to_task_ins_list(instructions)

        push_task_ins_req = driver_pb2.PushTaskInsRequest(task_ins_list=task_ins_list)

        # ---------------------------------------------------------------------- Driver SDK
        push_task_ins_res: driver_pb2.PushTaskInsResponse = driver.push_task_ins(
            req=push_task_ins_req
        )
        # ---------------------------------------------------------------------- Driver SDK

        time.sleep(sleep_time)

        # Wait for results, ignore empty task_ids
        start_time = time.time()
        task_ids: List[str] = [
            task_id for task_id in push_task_ins_res.task_ids if task_id != ""
        ]
        all_task_res: List[task_pb2.TaskRes] = []
        while True:
            if time.time() - start_time >= time_out:
                break
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

            time.sleep(sleep_time)

            all_task_res += task_res_list
            if len(all_task_res) == len(task_ids):
                break

        # Collect correct results
        node_messages = task_res_list_to_task_dict(
            [res for res in all_task_res if res.task.HasField("sa")]
        )
    workflow.close()

    # Slow down the start of the next round
    time.sleep(sleep_time)

# -------------------------------------------------------------------------- Driver SDK
driver.disconnect()
# -------------------------------------------------------------------------- Driver SDK
print("Driver disconnected")
