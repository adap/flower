"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

from pathlib import Path
from typing import List, Tuple

import flwr as fl
import hydra
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Metrics
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedrep.utils import (
    get_client_fn,
    get_create_model_fn,
    plot_metric_from_history,
    save_results_as_pickle,
    set_client_state_save_path,
    set_client_strategy,
)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameterss
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # set client strategy
    cfg = set_client_strategy(cfg)

    # Create directory to store client states if it does not exist
    # Client state has subdirectories with the name of current time
    client_state_save_path = set_client_state_save_path()

    # Define your clients
    # Get client function
    client_fn = get_client_fn(config=cfg, client_state_save_path=client_state_save_path)

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
            _ = server_round
            return fit_config

        return fit_config_fn

    # get a function that will be used to construct the model
    create_model, split = get_create_model_fn(cfg)

    model = split(create_model())

    def evaluate_metrics_aggregation_fn(
        eval_metrics: List[Tuple[int, Metrics]]
    ) -> Metrics:
        weights, accuracies = [], []
        for num_examples, metric in eval_metrics:
            weights.append(num_examples)
            accuracies.append(metric["accuracy"] * num_examples)
        accuracy = sum(accuracies) / sum(weights)  # type: ignore[arg-type]
        return {"accuracy": accuracy}

    # Define your strategy
    strategy = instantiate(
        cfg.strategy,
        initial_parameters=ndarrays_to_parameters(model.get_parameters()),
        on_fit_config_fn=get_on_fit_config(),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Save your results
    save_path = Path(HydraConfig.get().runtime.output_dir)

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path)
    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_local_epochs}"
        f"_R={cfg.num_rounds}"
        f"_lr={cfg.learning_rate}"
    )

    plot_metric_from_history(history, save_path, (file_suffix))


if __name__ == "__main__":
    main()
