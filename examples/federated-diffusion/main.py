from client import gen_client_fn
from strategy import ClientManager, SaveModelAndMetricsStrategy, trainconfig
from config import parse_args
import flwr as fl


if __name__ == "__main__":
    args = parse_args()
    client_manager = ClientManager()

    strategy = SaveModelAndMetricsStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=args.num_clients,  # Never sample less than 10 clients for training
        min_evaluate_clients=args.num_clients,  # Never sample less than 5 clients for evaluation
        min_available_clients=args.num_clients,  # Wait until all 10 clients are available
        on_fit_config_fn=trainconfig,
        on_evaluate_config_fn=trainconfig,
        client_manager=client_manager,
        args=args,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=gen_client_fn(args),
        num_clients=args.num_clients,
        # client_resources={"num_cpus": 10, "num_gpus":1},
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
