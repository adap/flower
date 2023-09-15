import flwr as fl
import hydra
from client import generate_client_fn
from dataset import gen_random_loaders
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Run the baseline.
    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    # print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, validationloaders, testloaders = gen_random_loaders(
        cfg.dataset.data,
        cfg.dataset.root,
        cfg.client.num_nodes,
        cfg.client.batch_size,
        cfg.client.num_classes_per_node,
    )

    # prepare function that will be used to spawn each client
    client_fn = generate_client_fn(trainloaders, testloaders, cfg)

    # instantiate strategy according to config
    strategy = instantiate(cfg.strategy, cfg)

    # Start simulation

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.client.num_nodes,
        config=fl.server.ServerConfig(num_rounds=cfg.client.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
