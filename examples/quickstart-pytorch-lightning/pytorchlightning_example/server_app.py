"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from pytorchlightning_example.task import LitAutoEncoder, get_parameters


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Convert model parameters to flwr.common.Parameters
    ndarrays = get_parameters(LitAutoEncoder())
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        initial_parameters=global_model_init,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
