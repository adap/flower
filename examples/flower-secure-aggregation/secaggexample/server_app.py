from typing import List, Tuple

from secaggexample.task import IS_DEMO, Net, get_weights
from secaggexample.workflow_with_log import SecAggPlusWorkflowWithLogs

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)


# Flower ServerApp
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Define strategy
    strategy = FedAvg(
        # Select all available clients
        fraction_fit=1.0,
        # Disable evaluation in demo
        fraction_evaluate=0.0 if IS_DEMO else context.run_config["fraction-evaluate"],
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    # Construct the LegacyContext
    num_rounds = int(context.run_config["num-server-rounds"])
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Create fit workflow
    if IS_DEMO:
        fit_workflow = SecAggPlusWorkflowWithLogs(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            timeout=context.run_config["timeout"],
        )
    else:
        fit_workflow = SecAggPlusWorkflow(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
        )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute
    workflow(driver, context)