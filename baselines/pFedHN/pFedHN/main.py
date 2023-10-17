"""Main script for pFedHN."""

import flwr as fl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pFedHN.client import generate_client_fn
from pFedHN.dataset import gen_random_loaders
from pFedHN.models import LocalLayer
from pFedHN.server import pFedHNServer
from pFedHN.utils import set_seed, get_device

from flwr.server.client_manager import SimpleClientManager
import torch

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    set_seed(42)

    # partition dataset and get dataloaders
    # pylint: disable=unbalanced-tuple-unpacking
    trainloaders, valloaders, testloaders = gen_random_loaders(
        cfg.dataset.data,
        cfg.dataset.root,
        cfg.client.num_nodes,
        cfg.client.batch_size,
        cfg.client.num_classes_per_node,
    )

    if cfg.model.variant == "pFedHNPC":
        node_local_layers = [LocalLayer(n_input=84, n_output=cfg.model.out_dim).to(get_device()) for _ in range(cfg.client.num_nodes)]
        node_local_optimizers = [torch.optim.SGD(node_local_layers[i].parameters(), lr=cfg.model.inner_lr, momentum=0.9, weight_decay=cfg.model.we_dec) for i in range(cfg.client.num_nodes)]
        client_fn = generate_client_fn(trainloaders, testloaders,valloaders, cfg,local_layers=node_local_layers,local_optims=node_local_optimizers,local=True)
    else:
        # prepare function that will be used to spawn each client
        client_fn = generate_client_fn(trainloaders, testloaders,valloaders, cfg,None,None,False)

    # instantiate strategy according to config
    strategy = instantiate(cfg.strategy, cfg)

    # Start simulation

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.client.num_nodes,
        server = pFedHNServer(client_manager=SimpleClientManager(),strategy=strategy,cfg=cfg),
        config=fl.server.ServerConfig(num_rounds=cfg.client.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
