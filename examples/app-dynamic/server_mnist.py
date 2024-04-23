from flwr.server import ServerApp, ServerConfig

from strategy import CustomFedAvg
from task import weighted_average

# Instantiate strategy
strategy = CustomFedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    dataset_name="mnist",
    net_name="mnist_net"
)

# Instantiate config
config = ServerConfig(num_rounds=3)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
