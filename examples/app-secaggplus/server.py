import flwr as fl
from flwr.common import Context
from flwr.server import Driver, LegacyContext
from flwr.server.workflow import SecAggPlusWorkflow
from workflow_with_log import SecAggPlusWorkflowWithLogs


# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=5,
    fit_metrics_aggregation_fn=lambda _: {},  # No metrics aggregation
)


# Run via `flower-server-app server_workflow:app`
app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Construct the LegacyContext
    context = LegacyContext(
        state=context.state,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Create the workflow
    workflow = fl.server.workflow.DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflowWithLogs(
            num_shares=3,
            reconstruction_threshold=2,
            timeout=5,
        )
        # # For real-world applications, use the following code instead
        # fit_workflow=SecAggPlusWorkflow(
        #     num_shares=<number of shares>,
        #     reconstruction_threshold=<reconstruction threshold>,
        # )
    )

    # Execute
    workflow(driver, context)
