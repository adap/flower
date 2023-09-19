# Copyright 2022 Adap GmbH. All Rights Reserved.
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
"""Default workflow factories."""


import timeit
from dataclasses import dataclass
from logging import DEBUG, INFO
from typing import Callable, Dict, Generator, Optional, Tuple

from flwr.common import serde
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns, Parameters, Scalar
from flwr.driver.task_utils import wrap_server_message_in_task
from flwr.proto.task_pb2 import Task
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy.strategy import Strategy


@dataclass
class WorkflowState:
    """State of the workflow."""

    num_rounds: int
    current_round: int
    strategy: Strategy
    parameters: Parameters
    client_manager: ClientManager
    history: History


FlowerWorkflow = Generator[Dict[ClientProxy, Task], Dict[ClientProxy, Task], None]
FlowerWorkflowFactory = Callable[[WorkflowState], FlowerWorkflow]


class FLWorkflowFactory:
    """Default FL workflow factory in Flower."""

    def __init__(
        self,
        fit_workflow_factory: Optional[FlowerWorkflowFactory] = None,
        evaluate_workflow_factory: Optional[FlowerWorkflowFactory] = None,
    ):
        self.fit_workflow_factory = (
            fit_workflow_factory
            if fit_workflow_factory is not None
            else default_fit_workflow_factory
        )
        self.evaluate_workflow_factory = (
            evaluate_workflow_factory
            if evaluate_workflow_factory is not None
            else default_evaluate_workflow_factory
        )

    def __call__(self, state: WorkflowState) -> FlowerWorkflow:
        """Create the workflow."""
        # Initialize parameters
        log(INFO, "Initializing global parameters")
        parameters = state.strategy.initialize_parameters(
            client_manager=state.client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            state.parameters = parameters
        # Get initial parameters from one of the clients
        else:
            log(INFO, "Requesting initial parameters from one random client")
            random_client = state.client_manager.sample(1)[0]
            # Send GetParametersIns and get the response
            node_responses: Dict[ClientProxy, Task] = yield {
                random_client: wrap_server_message_in_task(GetParametersIns(config={}))
            }
            get_parameters_res = serde.get_parameters_res_from_proto(
                node_responses[random_client].legacy_client_message.get_parameters_res
            )
            log(INFO, "Received initial parameters from one random client")
            state.parameters = get_parameters_res.parameters
        log(INFO, "Evaluating initial parameters")
        res = state.strategy.evaluate(0, parameters=state.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            state.history.add_loss_centralized(server_round=0, loss=res[0])
            state.history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, state.num_rounds + 1):
            state.current_round = current_round

            # Fit round
            yield from self.fit_workflow_factory(state)

            # Centralized evaluation
            res_cen = state.strategy.evaluate(
                current_round, parameters=state.parameters
            )
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
                state.history.add_loss_centralized(
                    server_round=current_round, loss=loss_cen
                )
                state.history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate round
            yield from self.evaluate_workflow_factory(state)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)


def default_fit_workflow_factory(state: WorkflowState) -> FlowerWorkflow:
    """Create the default workflow of a single fit round."""
    # Get clients and their respective instructions from strategy
    client_instructions = state.strategy.configure_fit(
        server_round=state.current_round,
        parameters=state.parameters,
        client_manager=state.client_manager,
    )

    if not client_instructions:
        log(INFO, "fit_round %s: no clients selected, cancel", state.current_round)
        return
    log(
        DEBUG,
        "fit_round %s: strategy sampled %s clients (out of %s)",
        state.current_round,
        len(client_instructions),
        state.client_manager.num_available(),
    )

    # Send instructions to clients and
    # collect `fit` results from all clients participating in this round
    node_responses = yield {
        proxy: wrap_server_message_in_task(fit_ins)
        for proxy, fit_ins in client_instructions
    }

    # No exception/failure handling currently
    log(
        DEBUG,
        "fit_round %s received %s results and %s failures",
        state.current_round,
        len(node_responses),
        0,
    )

    # Aggregate training results
    results = [
        (proxy, serde.fit_res_from_proto(res.legacy_client_message.fit_res))
        for proxy, res in node_responses.items()
    ]
    aggregated_result: Tuple[
        Optional[Parameters],
        Dict[str, Scalar],
    ] = state.strategy.aggregate_fit(state.current_round, results, [])
    parameters_aggregated, metrics_aggregated = aggregated_result

    # Update the parameters and write history
    if parameters_aggregated:
        state.parameters = parameters_aggregated
    state.history.add_metrics_distributed_fit(
        server_round=state.current_round, metrics=metrics_aggregated
    )
    return


def default_evaluate_workflow_factory(state: WorkflowState) -> FlowerWorkflow:
    """Create the default workflow of a single evaluate round."""
    # Get clients and their respective instructions from strategy
    client_instructions = state.strategy.configure_evaluate(
        server_round=state.current_round,
        parameters=state.parameters,
        client_manager=state.client_manager,
    )
    if not client_instructions:
        log(INFO, "evaluate_round %s: no clients selected, cancel", state.current_round)
        return
    log(
        DEBUG,
        "evaluate_round %s: strategy sampled %s clients (out of %s)",
        state.current_round,
        len(client_instructions),
        state.client_manager.num_available(),
    )

    # Send instructions to clients and
    # collect `evaluate` results from all clients participating in this round
    node_responses = yield {
        proxy: wrap_server_message_in_task(evaluate_ins)
        for proxy, evaluate_ins in client_instructions
    }
    # No exception/failure handling currently
    log(
        DEBUG,
        "evaluate_round %s received %s results and %s failures",
        state.current_round,
        len(node_responses),
        0,
    )

    # Aggregate the evaluation results
    results = [
        (proxy, serde.evaluate_res_from_proto(res.legacy_client_message.evaluate_res))
        for proxy, res in node_responses.items()
    ]
    aggregated_result: Tuple[
        Optional[float],
        Dict[str, Scalar],
    ] = state.strategy.aggregate_evaluate(state.current_round, results, [])

    loss_aggregated, metrics_aggregated = aggregated_result

    # Write history
    if loss_aggregated is not None:
        state.history.add_loss_distributed(
            server_round=state.current_round, loss=loss_aggregated
        )
        state.history.add_metrics_distributed(
            server_round=state.current_round, metrics=metrics_aggregated
        )
    return
