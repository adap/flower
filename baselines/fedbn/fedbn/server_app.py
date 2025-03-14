"""fedbn: A Flower Baseline."""

import json

from fedbn.model import CNNModel
from fedbn.server import ResultsSaverServer, save_results_and_config
from fedbn.strategy import get_metrics_aggregation_fn, get_on_fit_config
from fedbn.utils import context_to_easydict
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
    # Read from config
    print("### BEGIN: RUN CONFIG ####")
    configs = context_to_easydict(context)
    run_config = configs.run_config
    print(json.dumps(run_config, indent=4))
    print("### END: RUN CONFIG ####")
    num_rounds = context.run_config["num-server-rounds"]

    def get_weights(net):
        """Extract model parameters as numpy arrays from state_dict."""
        if run_config.algorithm_name == "FedAvg":
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
        return [
            val.cpu().numpy()
            for name, val in net.state_dict().items()
            if "bn" not in name
        ]

    ndarrays = get_weights(CNNModel(num_classes=run_config.num_classes))
    parameters = ndarrays_to_parameters(ndarrays)
    # Define Strategy
    strategy = FedAvg(
        fraction_fit=run_config.fraction_fit,
        fraction_evaluate=run_config.fraction_evaluate,
        min_available_clients=run_config.num_clients,
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
