from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from workflow_with_log import SecAggPlusWorkflowWithLogs


# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=5,
)


# Flower ServerApp
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Construct the LegacyContext
    context = LegacyContext(
        state=context.state,
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Create the workflow
    workflow = DefaultWorkflow(
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
