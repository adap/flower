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
"""Legacy default workflows."""


import timeit
from logging import DEBUG, INFO
from typing import Optional, cast

import flwr.common.recordset_compat as compat
from flwr.common import Context, GetParametersIns, log
from flwr.common.constant import (
    MESSAGE_TYPE_EVALUATE,
    MESSAGE_TYPE_FIT,
    MESSAGE_TYPE_GET_PARAMETERS,
)

from ..compat.legacy_context import LegacyContext
from ..driver import Driver
from ..typing import Workflow

KEY_CURRENT_ROUND = "current_round"
CONFIGS_RECORD_KEY = "config"
PARAMS_RECORD_KEY = "parameters"


class DefaultWorkflow:
    """Default workflow in Flower."""

    def __init__(
        self,
        fit_workflow: Optional[Workflow] = None,
        evaluate_workflow: Optional[Workflow] = None,
    ) -> None:
        if fit_workflow is None:
            fit_workflow = default_fit_workflow
        if evaluate_workflow is None:
            evaluate_workflow = default_evaluate_workflow
        self.fit_workflow: Workflow = fit_workflow
        self.evaluate_workflow: Workflow = evaluate_workflow

    def __call__(self, driver: Driver, context: Context) -> None:
        """Execute the workflow."""
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )

        # Initialize parameters
        default_init_params_workflow(driver, context)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, context.config.num_rounds + 1):
            context.state.configs_records[CONFIGS_RECORD_KEY][
                KEY_CURRENT_ROUND
            ] = current_round

            # Fit round
            self.fit_workflow(driver, context)

            # Centralized evaluation
            parameters = compat.parametersrecord_to_parameters(
                record=context.state.parameters_records[PARAMS_RECORD_KEY],
                keep_input=True,
            )
            res_cen = context.strategy.evaluate(current_round, parameters=parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                context.history.add_loss_centralized(
                    server_round=current_round, loss=loss_cen
                )
                context.history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate round
            self.evaluate_workflow(driver, context)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        

def update_client_manager(
    driver: GrpcDriver,
    client_manager: ClientManager,
    lock: threading.Lock,
    f_stop: threading.Event,
) -> None:
    """Update the nodes list in the client manager.

    This function periodically communicates with the associated driver to get all
    node_ids. Each node_id is then converted into a `DriverClientProxy` instance
    and stored in the `registered_nodes` dictionary with node_id as key.

    New nodes will be added to the ClientManager via `client_manager.register()`,
    and dead nodes will be removed from the ClientManager via
    `client_manager.unregister()`.
    """
    # Request for run_id
    run_id = driver.create_run(
        driver_pb2.CreateRunRequest()  # pylint: disable=E1101
    ).run_id

    # Loop until the driver is disconnected
    registered_nodes: Dict[int, DriverClientProxy] = {}
    while not f_stop.is_set():
        with lock:
            # End the while loop if the driver is disconnected
            if driver.stub is None:
                break
            get_nodes_res = driver.get_nodes(
                req=driver_pb2.GetNodesRequest(run_id=run_id)  # pylint: disable=E1101
            )
        all_node_ids = {node.node_id for node in get_nodes_res.nodes}
        dead_nodes = set(registered_nodes).difference(all_node_ids)
        new_nodes = all_node_ids.difference(registered_nodes)

        # Unregister dead nodes
        for node_id in dead_nodes:
            client_proxy = registered_nodes[node_id]
            client_manager.unregister(client_proxy)
            del registered_nodes[node_id]

        # Register new nodes
        for node_id in new_nodes:
            client_proxy = DriverClientProxy(
                node_id=node_id,
                driver=driver,
                anonymous=False,
                run_id=run_id,
            )
            if client_manager.register(client_proxy):
                registered_nodes[node_id] = client_proxy
            else:
                raise RuntimeError("Could not register node.")

        # Sleep for 3 seconds
        time.sleep(3)



def default_init_params_workflow(driver: Driver, context: Context) -> None:
    """Execute the default workflow for parameters initialization."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")

    log(INFO, "Initializing global parameters")
    parameters = context.strategy.initialize_parameters(
        client_manager=context.client_manager
    )
    if parameters is not None:
        log(INFO, "Using initial parameters provided by strategy")
        paramsrecord = compat.parameters_to_parametersrecord(
            parameters, keep_input=True
        )

    # Get initial parameters from one of the clients
    else:
        log(INFO, "Requesting initial parameters from one random client")
        random_client = context.client_manager.sample(1)[0]
        # Send GetParametersIns and get the response
        content = compat.getparametersins_to_recordset(GetParametersIns({}))
        messages = driver.send_and_receive(
            [
                driver.create_message(
                    content=content,
                    message_type=MESSAGE_TYPE_GET_PARAMETERS,
                    dst_node_id=random_client.node_id,
                    group_id="",
                    ttl="",
                )
            ]
        )
        log(INFO, "Received initial parameters from one random client")
        msg = list(messages)[0]
        paramsrecord = next(iter(msg.content.parameters_records.values()))

    context.state.parameters_records[PARAMS_RECORD_KEY] = paramsrecord

    # Evaluate initial parameters
    log(INFO, "Evaluating initial parameters")
    parameters = compat.parametersrecord_to_parameters(paramsrecord, keep_input=True)
    res = context.strategy.evaluate(0, parameters=parameters)
    if res is not None:
        log(
            INFO,
            "initial parameters (loss, other metrics): %s, %s",
            res[0],
            res[1],
        )
        context.history.add_loss_centralized(server_round=0, loss=res[0])
        context.history.add_metrics_centralized(server_round=0, metrics=res[1])


def default_evaluate_workflow(driver: Driver, context: Context) -> None:
    """Execute the default workflow for a single evaluate round."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")

    # Get current_round and parameters
    current_round = cast(
        int, context.state.configs_records[CONFIGS_RECORD_KEY][KEY_CURRENT_ROUND]
    )
    parametersrecord = context.state.parameters_records[PARAMS_RECORD_KEY]
    parameters = compat.parametersrecord_to_parameters(
        parametersrecord, keep_input=True
    )

    # Get clients and their respective instructions from strategy
    client_instructions = context.strategy.configure_evaluate(
        server_round=current_round,
        parameters=parameters,
        client_manager=context.client_manager,
    )
    if not client_instructions:
        log(INFO, "evaluate_round %s: no clients selected, cancel", current_round)
        return
    log(
        DEBUG,
        "evaluate_round %s: strategy sampled %s clients (out of %s)",
        current_round,
        len(client_instructions),
        context.client_manager.num_available(),
    )

    # Build dictionary mapping node_id to ClientProxy
    node_id_to_proxy = {proxy.node_id: proxy for proxy, _ in client_instructions}

    # Build out messages
    out_messages = [
        driver.create_message(
            content=compat.evaluateins_to_recordset(evalins, True),
            message_type=MESSAGE_TYPE_EVALUATE,
            dst_node_id=proxy.node_id,
            group_id="",
            ttl="",
        )
        for proxy, evalins in client_instructions
    ]

    # Send instructions to clients and
    # collect `evaluate` results from all clients participating in this round
    messages = list(driver.send_and_receive(out_messages))
    del out_messages

    # No exception/failure handling currently
    log(
        DEBUG,
        "evaluate_round %s received %s results and %s failures",
        current_round,
        len(messages),
        0,
    )

    # Aggregate the evaluation results
    results = [
        (
            node_id_to_proxy[msg.metadata.src_node_id],
            compat.recordset_to_evaluateres(msg.content),
        )
        for msg in messages
    ]
    aggregated_result = context.strategy.aggregate_evaluate(current_round, results, [])

    loss_aggregated, metrics_aggregated = aggregated_result

    # Write history
    if loss_aggregated is not None:
        context.history.add_loss_distributed(
            server_round=current_round, loss=loss_aggregated
        )
        context.history.add_metrics_distributed(
            server_round=current_round, metrics=metrics_aggregated
        )
