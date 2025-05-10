"""floco: A Flower Baseline."""

import json
import os
import time
from typing import Dict, Optional, Tuple, Union

import torch

from flwr.common import (
    Context,
    NDArrays,
    Scalar,
    bytes_to_ndarray,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .dataset import get_testloader
from .model import Net, SimplexModel, get_weights, set_weights, test
from .strategy import CustomFedAvg, Floco

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

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


def weighted_average(metrics):
    """Aggregate metrics by computing a weighted average of local accuracies."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    fed_acc = sum(accuracies) / sum(examples)
    fed_loss = sum(losses) / sum(examples)
    write_res(
        {"federated_evaluate_accuracy": fed_acc, "federated_evaluate_loss": fed_loss}
    )
    return {"federated_evaluate_accuracy": fed_acc}


def server_evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
    context: Context,
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate the current global model on the test set."""
    _ = server_round
    if context.run_config["algorithm"] == "Floco":
        server_model = SimplexModel(endpoints=context.run_config["endpoints"], seed=0)
        set_weights(server_model, parameters)
        server_model.training = False
        center_value = config.get("center")
        if isinstance(center_value, bytes):
            center = bytes_to_ndarray(center_value)
        else:
            center = None
        radius_value = config.get("radius")
        if isinstance(radius_value, (int, float)):
            radius = float(radius_value)
        else:
            radius = None
        server_model.subregion_parameters = (
            (center, radius) if center is not None and radius is not None else None
        )
    else:
        server_model = Net(seed=0)
        set_weights(server_model, parameters)
    dataset = str(context.run_config["dataset"])

    testloader = get_testloader(dataset)

    loss, accuracy = test(server_model, testloader, DEVICE)
    write_res({"centralized_loss": loss, "centralized_accuracy": accuracy})
    return loss, {"centralized_accuracy": accuracy}


def get_strategy(context: Context) -> Union[CustomFedAvg, Floco]:
    """Get the strategy."""
    fraction_fit = float(context.run_config["fraction-fit"])
    seed = int(context.run_config["seed"])

    if context.run_config["algorithm"] == "FedAvg":
        init_model = Net(seed=seed)
        init_parameters = ndarrays_to_parameters(get_weights(init_model))
        return CustomFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=init_parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=server_evaluate,
            context=context,
        )
    if context.run_config["algorithm"] == "Floco":
        tau = int(context.run_config["tau"])
        rho = float(context.run_config["rho"])
        endpoints = int(context.run_config["endpoints"])
        init_model = SimplexModel(endpoints=endpoints, seed=seed)
        init_parameters = ndarrays_to_parameters(get_weights(init_model))
        return Floco(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=init_parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=server_evaluate,
            context=context,
            tau=tau,
            rho=rho,
            endpoints=endpoints,
        )
    raise ValueError("Algorithm not implemented")


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    config_json_file(context)
    num_rounds = int(context.run_config["num-server-rounds"])
    strategy = get_strategy(context)
    config = ServerConfig(num_rounds=int(num_rounds))
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
