"""fedprox: A Flower Baseline."""

import json
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import NDArrays, Scalar
from flwr.server import (
    ServerApp,
    ServerAppComponents,
    ServerConfig,
    SimpleClientManager,
)

from .dataset import prepare_test_loader
from .model import get_weights, instantiate_model, set_weights, test
from .server import ResultsSaverServer, history_saver
from .strategy import FedAvgWithStragglerDrop
from .utils import context_to_easydict


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    run_config: dict,
) -> Callable[
    [int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, dict[str, Scalar]] | None:
        # pylint: disable=unused-argument
        """Use the entire MNIST test set for evaluation."""
        net = instantiate_model(run_config)
        set_weights(net, parameters_ndarrays)
        net.to(device)

        # We could compile the model here but we are not going to do it because
        # running test() is so lightweight that the overhead of compiling the model
        # negate any potential speedup. Please note this is specific to the model and
        # dataset used in this baseline. In general, compiling the model is worth it

        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Do weighted average of accuracy metric."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    print("### BEGIN: Experiment Config ###")
    configs = context_to_easydict(context)
    run_config = configs.run_config  # pylint: disable=E1101
    print(json.dumps(run_config, indent=4))
    print("### END: Experiment Config ###")
    # Initialize model parameters
    ndarrays = get_weights(instantiate_model(run_config))
    parameters = ndarrays_to_parameters(ndarrays)

    # get a function that will be used to construct the config that the client's
    # fit() method will receive

    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            return {"current_round": server_round}

        return fit_config_fn

    device = torch.device("cpu")
    testloader = prepare_test_loader(
        configs.run_config.dataset  # pylint: disable=E1101
    )  # for server-side evaluation
    evaluate_fn = gen_evaluate_fn(
        testloader,
        device=device,
        run_config=configs.run_config,  # pylint: disable=E1101
    )
    # Define strategy
    strategy = FedAvgWithStragglerDrop(
        fraction_fit=float(run_config.algorithm.fraction_fit),
        fraction_evaluate=run_config.algorithm.fraction_evaluate,
        min_available_clients=run_config.algorithm.min_available_clients,
        initial_parameters=parameters,
        on_fit_config_fn=get_on_fit_config(),
        evaluate_fn=evaluate_fn,
    )
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=history_saver,
        run_config=run_config,
    )
    config = ServerConfig(num_rounds=int(run_config.algorithm.num_server_rounds))
    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
