# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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

from logging import INFO
from flwr.common import log
from typing import Optional
from ..typing import Workflow
from ..compat.legacy_context import LegacyContext
from ..driver import Driver
from flwr.common import Context, Parameters
import flwr.common.recordset_compat as compat


KEY_CURRENT_ROUND = "current_round"
CONFIGS_RECORD_KEY = "config"
PARAMS_RECORD_KEY = "parameters"


class DefaultWorkflow:
    """Default FL workflow factory in Flower."""

    def __init__(
        self,
        fit_workflow: Optional[Workflow] = None,
        evaluate_workflow: Optional[Workflow] = None,
    ):
        if fit_workflow is None:
            fit_workflow = default_fit_workflow
        if evaluate_workflow is None:
            evaluate_workflow = default_evaluate_workflow
        self.fit_workflow: Workflow = fit_workflow
        self.evaluate_workflow: Workflow = evaluate_workflow

    def __call__(self, driver: Driver, context: Context) -> None:
        """Create the workflow."""
        if not isinstance(context, LegacyContext):
            raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")
        
        # Initialize parameters
        default_init_params_workflow(driver, context)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, context.config.num_rounds + 1):
            context.state.configs[CONFIGS_RECORD_KEY][KEY_CURRENT_ROUND] = current_round

            # Fit round
            self.fit_workflow(driver, context)

            # Centralized evaluation
            parameters = compat.parametersrecord_to_parameters(
                record=context.state.parameters[PARAMS_RECORD_KEY],
                keep_input=True,
            )
            res_cen = context.strategy.evaluate(
                current_round, parameters=parameters
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


def default_init_params_workflow(driver: Driver, context: Context) -> None:
    """Create the default workflow for parameters initialization."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")
    
    log(INFO, "Initializing global parameters")
    parameters = context.strategy.initialize_parameters(
        client_manager=context.client_manager
    )
    if parameters is not None:
        log(INFO, "Using initial parameters provided by strategy")
        state.parameters = parameters
    # Get initial parameters from one of the clients
    else:
        log(INFO, "Requesting initial parameters from one random client")
        random_client = state.client_manager.sample(1)[0]
        # Send GetParametersIns and get the response
        node_responses = yield {
            random_client.node_id: wrap_server_message_in_task(
                GetParametersIns(config={})
            )
        }
        get_parameters_res = serde.get_parameters_res_from_proto(
            node_responses[
                random_client.node_id
            ].legacy_client_message.get_parameters_res
        )
        log(INFO, "Received initial parameters from one random client")
        state.parameters = get_parameters_res.parameters

    # Evaluate initial parameters
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


def default_fit_workflow(state: WorkflowState) -> FlowerWorkflow:
    """Create the default workflow for a single fit round."""
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

    # Build dictionary mapping node_id to ClientProxy
    node_id_to_proxy = {proxy.node_id: proxy for proxy, _ in client_instructions}

    # Send instructions to clients and
    # collect `fit` results from all clients participating in this round
    node_responses = yield {
        proxy.node_id: wrap_server_message_in_task(fit_ins)
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
        (
            node_id_to_proxy[node_id],
            serde.fit_res_from_proto(res.legacy_client_message.fit_res),
        )
        for node_id, res in node_responses.items()
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


def default_evaluate_workflow(state: WorkflowState) -> FlowerWorkflow:
    """Create the default workflow for a single evaluate round."""
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

    # Build dictionary mapping node_id to ClientProxy
    node_id_to_proxy = {proxy.node_id: proxy for proxy, _ in client_instructions}

    # Send instructions to clients and
    # collect `evaluate` results from all clients participating in this round
    node_responses = yield {
        proxy.node_id: wrap_server_message_in_task(evaluate_ins)
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
        (
            node_id_to_proxy[node_id],
            serde.evaluate_res_from_proto(res.legacy_client_message.evaluate_res),
        )
        for node_id, res in node_responses.items()
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