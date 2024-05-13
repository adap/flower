import flwr as fl
from pathlib import Path

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


if __name__ == "__main__":
    hist = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        certificates=(
            Path("certificates/ca.crt").read_bytes(),
            Path("certificates/server.pem").read_bytes(),
            Path("certificates/server.key").read_bytes(),
        ),
    )

    assert (
        hist.losses_distributed[-1][1] == 0
        or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98
    )
