from typing import List
import random
import time

from flwr.driver import (
    Driver,
    GetClientsResponse,
    GetClientsRequest,
    Task,
    Result,
    CreateTasksRequest,
    CreateTasksResponse,
    GetResultsRequest,
    GetResultsResponse,
    TaskAssignment,
)
from flwr.common import ServerMessage, FitIns, ndarrays_to_parameters

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

    # Get a list of client ID's from the server
    get_clients_req = GetClientsRequest()

    # ---------------------------------------------------------------------- Driver SDK
    get_clients_res: GetClientsResponse = driver.get_clients(req=get_clients_req)
    # ---------------------------------------------------------------------- Driver SDK

    # Sample three clients
    all_client_ids: List[int] = get_clients_res.client_ids
    print(f"Got {len(all_client_ids)} client IDs")
    sampled_client_ids: List[int] = random.sample(all_client_ids, 3)
    print(f"Sampled {len(sampled_client_ids)} client IDs")

    time.sleep(sleep_time)

    # Schedule a task for all three clients
    fit_ins: FitIns = FitIns(parameters=parameters, config={})
    task = Task(task_id=123, legacy_server_message=ServerMessage(fit_ins=fit_ins))
    task_assignment: TaskAssignment = TaskAssignment(
        task=task, client_ids=sampled_client_ids
    )
    create_tasks_req = CreateTasksRequest(task_assignments=[task_assignment])

    # ---------------------------------------------------------------------- Driver SDK
    create_tasks_res: CreateTasksResponse = driver.create_tasks(req=create_tasks_req)
    # ---------------------------------------------------------------------- Driver SDK

    print(f"Scheduled {len(create_tasks_res.task_ids)} tasks")

    time.sleep(sleep_time)

    # Wait for results
    task_ids: List[int] = create_tasks_res.task_ids
    all_results: List[Result] = []
    while True:
        get_results_req = GetResultsRequest(task_ids=create_tasks_res.task_ids)

        # ------------------------------------------------------------------ Driver SDK
        get_results_res: GetResultsResponse = driver.get_results(req=get_results_req)
        # ------------------------------------------------------------------ Driver SDK

        results: List[Result] = get_results_res.results
        print(f"Got {len(get_results_res.results)} results")

        time.sleep(sleep_time)

        all_results += results
        if len(all_results) == len(task_ids):
            break

    # "Aggregate" results
    client_messages = [result.legacy_client_message for result in all_results]
    print(f"Received {len(client_messages)} results")

    time.sleep(sleep_time)

    # Repeat

# -------------------------------------------------------------------------- Driver SDK
driver.disconnect()
# -------------------------------------------------------------------------- Driver SDK
