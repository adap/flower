"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import flwr as fl
import hydra
from omegaconf import DictConfig, OmegaConf

from fedavgm.client import generate_client_fn
from fedavgm.dataset import partition, prepare_dataset
from fedavgm.models import create_model
from fedavgm.server import get_evaluate_fn, get_on_fit_config


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_dataset(
        cfg.dataset.fmnist
    )
    partitions = partition(
        x_train, y_train, cfg.num_clients, cfg.dataset.concentration, num_classes
    )

    print(f">>> [Model]: Num. Classes {num_classes} | Input shape: {input_shape}")

    # 3. Define your clients
    client_fn = generate_client_fn(
        partitions,
        input_shape,
        num_classes,
    )

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

    if cfg.fedavgm is True:
        fedavgm = fl.server.strategy.FedAvgM(
            min_available_clients=cfg.num_clients,
            fraction_fit=cfg.server.reporting_fraction,
            fraction_evaluate=cfg.server.reporting_fraction,
            on_fit_config_fn=get_on_fit_config(cfg.client),
            evaluate_fn=get_evaluate_fn(
                input_shape, num_classes, x_test, y_test, cfg.num_rounds
            ),  # server evaluation of the global model
            server_learning_rate=cfg.server.learning_rate,
            server_momentum=cfg.server.momentum,
            initial_parameters=fl.common.ndarrays_to_parameters(
                create_model(input_shape, num_classes).get_weights()
            ),
        )
        print(
            f">>> [Strategy] FedAvgM | Num. Clients: {cfg.num_clients} | \
                Fraction: {cfg.server.reporting_fraction}..."
        )
        fedavg = None
    else:
        fedavg = fl.server.strategy.FedAvg(
            min_available_clients=cfg.num_clients,
            fraction_fit=cfg.server.reporting_fraction,
            fraction_evaluate=cfg.server.reporting_fraction,
            on_fit_config_fn=get_on_fit_config(cfg.client),
            evaluate_fn=get_evaluate_fn(
                input_shape, num_classes, x_test, y_test, cfg.num_rounds
            ),  # server evaluation of the global model
        )
        print(
            f">>> [Strategy] FedAvg | Num. Clients: {cfg.num_clients} | \
                Fraction: {cfg.server.reporting_fraction}..."
        )
        fedavgm = None

    # 5. Start Simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=fedavgm or fedavg,
    )

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
    main()
