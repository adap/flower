import flwr as fl

# Define metric aggregation function
def agg(metrics):
    # Weigh accuracy of each client by number of examples used
    accuracies = [m["accuracy"] * n for m, n in metrics]
    examples = [n for _, n in metrics]

    # Aggregate and return custom metric
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=agg)

# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config={"num_rounds": 3},
    strategy=strategy,
)
