import flwr as fl
from flwr.server import LegacyContext

app = fl.server.ServerApp()


@app.main()
def main(driver, context) -> None:
    # Construct the LegacyContext
    context = LegacyContext(
        state=context.state,
        config=fl.server.ServerConfig(num_rounds=3),
    )

    # Create the workflow
    workflow = fl.server.workflow.DefaultWorkflow()

    # Execute
    workflow(driver, context)

    assert context.history.losses_distributed[-1][1] == 0
