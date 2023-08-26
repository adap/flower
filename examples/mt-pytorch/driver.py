from typing import List, Tuple
import random
import time

from flwr.driver import Driver
from flwr.common import (
    ServerMessage,
    FitIns,
    ndarrays_to_parameters,
    serde,
    parameters_to_ndarrays,
    ClientMessage,
    NDArrays,
    Code,
)
from flwr.proto import driver_pb2, task_pb2, node_pb2, transport_pb2
from flwr.server.strategy.aggregate import aggregate
from flwr.common import Metrics
from flwr.server import History
from flwr.common import serde
from task import Net, get_parameters, set_parameters


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
driver = Driver(driver_service_address="0.0.0.0:9091", certificates=None)
# -------------------------------------------------------------------------- Driver SDK

anonymous_client_nodes = False
num_client_nodes_per_round = 2
sleep_time = 1
num_rounds = 3
parameters = ndarrays_to_parameters(get_parameters(net=Net()))

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
            get_nodes_req = driver_pb2.GetNodesRequest()

            # ---------------------------------------------------------------------- Driver SDK
            get_nodes_res: driver_pb2.GetNodesResponse = driver.get_nodes(
                req=get_nodes_req
            )
            # ---------------------------------------------------------------------- Driver SDK

            all_node_ids: List[int] = get_nodes_res.node_ids
            print(f"Got {len(all_node_ids)} node IDs")

            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                sampled_node_ids = random.sample(
                    all_node_ids, num_client_nodes_per_round
                )
                break

            time.sleep(3)

    # Log sampled node IDs
    print(f"Sampled {len(sampled_node_ids)} node IDs: {sampled_node_ids}")
    time.sleep(sleep_time)

    # Schedule a task for all sampled nodes
    fit_ins: FitIns = FitIns(parameters=parameters, config={})
    server_message_proto: transport_pb2.ServerMessage = serde.server_message_to_proto(
        server_message=ServerMessage(fit_ins=fit_ins)
    )
    task_ins_list: List[task_pb2.TaskIns] = []
    for sampled_node_id in sampled_node_ids:
        new_task_ins = task_pb2.TaskIns(
            task_id="",  # Do not set, will be created and set by the DriverAPI
            group_id="",
            workload_id=workload_id,
            task=task_pb2.Task(
                producer=node_pb2.Node(
                    node_id=0,
                    anonymous=True,
                ),
                consumer=node_pb2.Node(
                    node_id=sampled_node_id,
                    anonymous=anonymous_client_nodes,
                    # Must be True if we're working with anonymous clients
                ),
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

    # Collect correct results
    node_messages: List[ClientMessage] = []
    for task_res in all_task_res:
        if task_res.task.HasField("legacy_client_message"):
            node_messages.append(task_res.task.legacy_client_message)
    print(f"Received {len(node_messages)} results")

    weights_results: List[Tuple[NDArrays, int]] = []
    metrics_results: List = []
    for node_message in node_messages:
        if not node_message.fit_res:
            continue
        fit_res = node_message.fit_res
        # Aggregate only if the status is OK
        if fit_res.status.code != Code.OK.value:
            continue
        weights_results.append(
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        )
        metrics_results.append(
            (fit_res.num_examples, serde.metrics_from_proto(fit_res.metrics))
        )

    # Aggregate parameters (FedAvg)
    parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
    parameters = parameters_aggregated

    # Aggregate metrics
    metrics_aggregated = weighted_average(metrics_results)
    history.add_metrics_distributed_fit(
        server_round=server_round, metrics=metrics_aggregated
    )
    print("Round ", server_round, " metrics: ", metrics_aggregated)

    # Slow down the start of the next round
    time.sleep(sleep_time)

print("app_fit: losses_distributed %s", str(history.losses_distributed))
print("app_fit: metrics_distributed_fit %s", str(history.metrics_distributed_fit))
print("app_fit: metrics_distributed %s", str(history.metrics_distributed))
print("app_fit: losses_centralized %s", str(history.losses_centralized))
print("app_fit: metrics_centralized %s", str(history.metrics_centralized))

# -------------------------------------------------------------------------- Driver SDK
driver.disconnect()
# -------------------------------------------------------------------------- Driver SDK
print("Driver disconnected")
