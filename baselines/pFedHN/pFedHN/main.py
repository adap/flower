import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import gen_random_loaders
from client import generate_client_fn

@hydra.main(config_path="conf",config_name="base",version_base=None)
def main(cfg:DictConfig):

    # print(OmegaConf.to_yaml(cfg))

    trainloaders, validationloaders, testloaders = gen_random_loaders(cfg.dataset.data,cfg.dataset.root,cfg.client.num_nodes,cfg.client.batch_size,cfg.client.num_classes_per_node)

    
    client_fn = generate_client_fn(trainloaders,testloaders,cfg.client.num_nodes)

    
    strategy = instantiate(
        cfg.strategy
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.client.num_nodes,
        config = fl.server.ServerConfig(num_rounds=cfg.client.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":

    main()