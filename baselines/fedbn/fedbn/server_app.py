"""fedbn: A Flower Baseline."""

import json

from fedbn.model import CNNModel
from fedbn.server import ResultsSaverServer, save_results_and_config
from fedbn.strategy import get_metrics_aggregation_fn, get_on_fit_config
from fedbn.utils import extract_weights
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import (
    ServerApp,
    ServerAppComponents,
    ServerConfig,
    SimpleClientManager,
)
from flwr.server.strategy import FedAvg


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from context
    print("### BEGIN: RUN CONFIG ####")
    run_config = context.run_config
    print(json.dumps(run_config, indent=4))
    print("### END: RUN CONFIG ####")
    num_rounds = context.run_config["num-server-rounds"]

    ndarrays = extract_weights(
        CNNModel(num_classes=run_config["num-classes"]),
        run_config["algorithm-name"],
    )
    parameters = ndarrays_to_parameters(ndarrays)
    # Define Strategy
    strategy = FedAvg(
        fraction_fit=float(run_config["fraction-fit"]),
        fraction_evaluate=float(run_config["fraction-evaluate"]),
        min_available_clients=int(run_config["num-clients"]),
        on_fit_config_fn=get_on_fit_config(),
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=get_metrics_aggregation_fn(),
        evaluate_metrics_aggregation_fn=get_metrics_aggregation_fn(),
    )
    config = ServerConfig(num_rounds=int(num_rounds))
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=save_results_and_config,
        run_config=run_config,
    )
    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
