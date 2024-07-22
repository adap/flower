from secaggexample.workflow_with_log import SecAggPlusWorkflowWithLogs

from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

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
    num_rounds = int(context.run_config["num_server_rounds"])
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Create the workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflowWithLogs(
            num_shares=int(context.run_config["num_shares"]),
            reconstruction_threshold=int(
                context.run_config["reconstruction_threshold"]
            ),
            timeout=float(context.run_config["timeout"]),
        )
        # # For real-world applications, use the following code instead
        # fit_workflow=SecAggPlusWorkflow(
        #     num_shares=int(context.run_config["num_shares"]),
        #     reconstruction_threshold=int(context.run_config["reconstruction_threshold"]),
        # )
    )

    # Execute
    workflow(driver, context)
