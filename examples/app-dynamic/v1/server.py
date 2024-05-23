from flwr.server import ServerApp, ServerConfig

from strategy import CustomFedAvg
from task import weighted_average

# Instantiate strategy
strategy_cifar = CustomFedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    dataset_name="cifar",
    net_name="cifar_net"
)
strategy_bigger_cifar = CustomFedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    dataset_name="cifar",
    net_name="bigger_cifar_net"
)
strategy_mnist = CustomFedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    dataset_name="mnist",
    net_name="mnist_net"
)

# Instantiate config
config = ServerConfig(num_rounds=2)

# Flower ServerApp
app_cifar = ServerApp(
    config=config,
    strategy=strategy_cifar,
)

app_mnist = ServerApp(
    config=config,
    strategy=strategy_mnist,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy_mnist,
    )
