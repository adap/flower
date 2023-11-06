import flwr as fl
from custom_strategy import custom_FedHeNN

num_rounds = 5
from model_mnist import Net0, Net1, Net2, Net3


def fit_config(rnd):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),  # number of local epochs
        "batch_size": str(32),
        "num_rounds": str(num_rounds),  # number of rounds
    }
    return config


def eval_config(rnd):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "num_rounds": str(num_rounds),  # number of rounds
    }
    return config


if __name__ == "__main__":

    Weights_init0 = [val.cpu().numpy() for _, val in Net0().state_dict().items()]
    Weights_init1 = [val.cpu().numpy() for _, val in Net1().state_dict().items()]
    Weights_init2 = [val.cpu().numpy() for _, val in Net2().state_dict().items()]
    Weights_init3 = [val.cpu().numpy() for _, val in Net3().state_dict().items()]

    # Define strategy
    strategy = custom_FedHeNN(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=4,
        min_eval_clients=4,
        min_available_clients=4,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        initial_parameters=[Weights_init0, Weights_init1, Weights_init2, Weights_init3],
        fit_metrics_aggregation_fn=True
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": num_rounds},
        strategy=strategy,
    )
