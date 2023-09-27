import flwr as fl

from client import client_fn
from config import PARAMS
from strategy import ClientManager, SaveModelAndMetricsStrategy, trainconfig


client_manager = ClientManager()

strategy = SaveModelAndMetricsStrategy(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=PARAMS.num_clients,  # Never sample less than 10 clients for training
    min_evaluate_clients=PARAMS.num_clients,  # Never sample less than 5 clients for evaluation
    min_available_clients=PARAMS.num_clients,  # Wait until all 10 clients are available
    on_fit_config_fn=trainconfig,
    on_evaluate_config_fn=trainconfig,
    client_manager=client_manager,
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=PARAMS.num_clients,
    # client_resources={"num_cpus": 10, "num_gpus":1},
    # client_resources={"num_gpus":1},
    config=fl.server.ServerConfig(num_rounds=PARAMS.num_rounds),
    strategy=strategy,
)
