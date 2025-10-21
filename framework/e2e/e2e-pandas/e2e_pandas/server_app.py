from e2e_pandas.strategy import FedAnalytics

import flwr as fl

app = fl.serverapp.ServerApp()


@app.main()
def main(grid, context):
    # Construct the LegacyContext
    context = fl.server.LegacyContext(
        context=context,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=FedAnalytics(),
    )

    # Create the workflow
    workflow = fl.server.workflow.DefaultWorkflow()

    # Execute
    workflow(grid, context)

    hist = context.history
    assert hist.metrics_centralized["Aggregated histograms"][1][1] == [
        "Length:",
        "18",
        "46",
        "28",
        "54",
        "32",
        "52",
        "36",
        "12",
        "10",
        "12",
        "Width:",
        "8",
        "14",
        "44",
        "48",
        "74",
        "62",
        "20",
        "22",
        "4",
        "4",
    ]


if __name__ == "__main__":
    hist = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=FedAnalytics(),
    )

    assert hist.metrics_centralized["Aggregated histograms"][1][1] == [
        "Length:",
        "18",
        "46",
        "28",
        "54",
        "32",
        "52",
        "36",
        "12",
        "10",
        "12",
        "Width:",
        "8",
        "14",
        "44",
        "48",
        "74",
        "62",
        "20",
        "22",
        "4",
        "4",
    ]
