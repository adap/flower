import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import flwr as fl

from vitexample.task import apply_eval_transforms
from vitexample.task import get_model, set_params, test, get_params

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg


def get_evaluate_fn(
    centralized_testset: Dataset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, parameters, config):
        """Use the entire Oxford Flowers-102 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_model()
        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_eval_transforms)

        testloader = DataLoader(testset, batch_size=128)
        # Run evaluation
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate


def server_fn(context: Context):

    # Define tested for central evaluation
    dataset = load_dataset("nelorth/oxford-flowers")
    test_set = dataset["test"]

    # Set initial global model
    ndarrays = get_params(get_model())
    init_parameters = ndarrays_to_parameters(ndarrays)

    # Configure the strategy
    strategy = FedAvg(
        fraction_fit=0.5,  # Sample 50% of available clients
        fraction_evaluate=0.0,  # No federated evaluation
        evaluate_fn=get_evaluate_fn(test_set),  # Global evaluation function
        initial_parameters=init_parameters,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
