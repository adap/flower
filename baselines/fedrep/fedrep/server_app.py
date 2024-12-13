"""fedrep: A Flower Baseline."""

import json
import os
import time
from typing import Dict, List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .utils import get_create_model_fn, get_server_strategy

RESULTS_FILE = "result-{}.json"


def config_json_file(context: Context) -> None:
    """Initialize the json file and write the run configurations."""
    # Initialize the execution results directory.
    res_save_path = "./results"
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)
    res_save_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Set the date and full path of the file to save the results.
    global RESULTS_FILE  # pylint: disable=global-statement
    RESULTS_FILE = RESULTS_FILE.format(res_save_name)
    RESULTS_FILE = f"{res_save_path}/{RESULTS_FILE}"
    data = {
        "run_config": dict(context.run_config.items()),
        "round_res": [],
    }
    with open(RESULTS_FILE, "w+", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)


def write_res(new_res: Dict[str, float]) -> None:
    """Load the json file, append result and re-write json collection."""
    with open(RESULTS_FILE, "r", encoding="UTF-8") as fin:
        data = json.load(fin)
    data["round_res"].append(new_res)

    # Write the updated data back to the JSON file
    with open(RESULTS_FILE, "w", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)


def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted metrics evaluation."""
    weights, accuracies, losses = [], [], []
    for num_examples, metric in eval_metrics:
        weights.append(num_examples)
        accuracies.append(float(metric["accuracy"]) * num_examples)
        losses.append(float(metric["loss"]) * num_examples)
    accuracy = sum(accuracies) / sum(weights)
    loss = sum(losses) / sum(weights)
    write_res({"accuracy": accuracy, "loss": loss})
    return {"accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    config_json_file(context)
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    create_model_fn, split_class = get_create_model_fn(context)
    net = split_class(create_model_fn())
    parameters = ndarrays_to_parameters(net.get_parameters())

    # Define strategy
    strategy = get_server_strategy(
        context=context, params=parameters, eval_fn=evaluate_metrics_aggregation_fn
    )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
