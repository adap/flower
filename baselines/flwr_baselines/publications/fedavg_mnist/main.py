"""Runs CNN federated learning for MNST dataset."""
import flwr as fl
import hydra
import torch
from omegaconf import DictConfig

from flwr_baselines.publications.fedavg_mnist import client, utils

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    client_fn, testloader = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        num_clients=cfg.num_clients,
        idd=cfg.idd,
        learning_rate=cfg.learning_rate,
    )

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.client_fraction,
        fraction_evaluate=cfg.client_fraction / 2,
        min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
        min_evaluate_clients=int(cfg.num_clients / 2),
        min_available_clients=cfg.num_clients,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    utils.plot_metric_from_history(history, cfg.plot_path, cfg.expected_maximum)


if __name__ == "__main__":
    main()
