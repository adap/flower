"""vitexample: A Flower / PyTorch app with Vision Transformers."""

from logging import INFO

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from vitexample.task import apply_eval_transforms
from vitexample.task import get_model, set_params, test, get_params

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg


def get_evaluate_fn(
    centralized_testset: Dataset,
    num_classes: int,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, parameters, config):
        """Use the entire Oxford Flowers-102 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Instantiate model and apply current global parameters
        model = get_model(num_classes)
        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_eval_transforms)

        testloader = DataLoader(testset, batch_size=128)
        # Run evaluation
        loss, accuracy = test(model, testloader, device=device)
        log(INFO, f"round: {server_round} -> acc: {accuracy:.4f}, loss: {loss: .4f}")

        return loss, {"accuracy": accuracy}

    return evaluate


def server_fn(context: Context):

    # Define tested for central evaluation
    dataset_name = context.run_config["dataset-name"]
    dataset = load_dataset(dataset_name)
    test_set = dataset["test"]

    # Set initial global model
    num_classes = context.run_config["num-classes"]
    ndarrays = get_params(get_model(num_classes))
    init_parameters = ndarrays_to_parameters(ndarrays)

    # Configure the strategy
    strategy = FedAvg(
        fraction_fit=0.5,  # Sample 50% of available clients
        fraction_evaluate=0.0,  # No federated evaluation
        evaluate_fn=get_evaluate_fn(
            test_set, num_classes
        ),  # Global evaluation function
        initial_parameters=init_parameters,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
