import flwr as fl

app = fl.server.ServerApp()


@app.main()
def main(driver, context):
    # Construct the LegacyContext
    context = fl.server.LegacyContext(
        state=context.state,
        config=fl.server.ServerConfig(num_rounds=3),
    )

    # Create the workflow
    workflow = fl.server.workflow.DefaultWorkflow()

    # Execute
    workflow(driver, context)

    hist = context.history
    assert (
        hist.losses_distributed[-1][1] == 0
        or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98
    )
