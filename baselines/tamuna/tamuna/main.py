"""Runs CNN federated learning for MNIST dataset."""

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tamuna import client, server, utils
from tamuna.dataset import load_datasets
from tamuna.utils import save_results_as_pickle


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, testloader = load_datasets(num_clients=cfg.num_clients)

    # prepare function that will be used to spawn each client
    client_fn = client.gen_client_fn(
        trainloaders=trainloaders,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
        client_device=cfg.client_device,
    )

    evaluate_fn = server.gen_evaluate_fn(
        testloader, device=cfg.server_device, model=cfg.model
    )

    epochs_per_round = np.random.geometric(p=cfg.p, size=cfg.num_rounds)
    strategy = instantiate(
        cfg.strategy,
        clients_per_round=cfg.clients_per_round,
        epochs_per_round=epochs_per_round,
        evaluate_fn=evaluate_fn,
    )

    # Start simulation
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

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"_C={cfg.num_clients}"
        f"_E={int(1 / cfg.p)}"
        f"_R={cfg.num_rounds}"
    )

    utils.plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )


if __name__ == "__main__":
    main()
