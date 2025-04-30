"""dasha: A Flower Baseline."""

import json
from typing import List, Tuple

from dasha.dataset import load_dataset, random_split
from dasha.dataset_preparation import find_pre_downloaded_or_download_dataset
from dasha.model import define_model
from dasha.server import ResultsSaverServer, save_results_and_config
from dasha.strategy import define_strategy
from dasha.utils import (
    _get_dataset_input_shape,
    get_parameters,
    reformat_config,
    set_seed,
)
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import (
    ServerApp,
    ServerAppComponents,
    ServerConfig,
    SimpleClientManager,
)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Do weighted average of accuracy metric."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [
        num_examples * float(m["accuracy"]) for num_examples, m in metrics
    ]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    print("### BEGIN: RUN CONFIG ####")
    cfg = reformat_config(context.run_config)
    print(json.dumps(cfg, indent=4))
    print("### END: RUN CONFIG ####")
    # Pre-download the dataset
    find_pre_downloaded_or_download_dataset(cfg)
    set_seed(seed=42)
    # Read from config
    num_rounds = cfg["num-server-rounds"]
    # Initialize model parameters with test dataset for shape.
    dataset = load_dataset(cfg)
    datasets = random_split(dataset, cfg["num-clients"])
    local_dataset = datasets[0]
    model = define_model(
        cfg["model"], input_shape=_get_dataset_input_shape(local_dataset)
    )
    ndarrays = get_parameters(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy_type = define_strategy(cfg["method"]["name"].lower())
    strategy = strategy_type(
        step_size=cfg["method"]["step-size"],
        num_clients=cfg["num-clients"],
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=int(num_rounds))
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=save_results_and_config,
        run_config=cfg,
    )
    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
