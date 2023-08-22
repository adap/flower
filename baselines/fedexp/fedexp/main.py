import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from fedexp import client, server
from fedexp.dataset import load_datasets
from fedexp.utils import seed_everything, get_parameters
import numpy as np


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    trainloaders, testloader = load_datasets(config=cfg.dataset_config,
                                             num_clients=cfg.num_clients,
                                             batch_size=cfg.batch_size,
                                             partition_equal=True)

    p = np.zeros(cfg.num_clients)
    for i in range(cfg.num_clients):
        p[i] = len(trainloaders[i])
    p /= np.sum(p)

    client_fn = client.gen_client_fn(trainloaders=trainloaders,
                                     model=cfg.model,
                                     num_epochs=cfg.num_epochs,
                                     args={"p": p},
                                     )

    evaluate_fn = server.gen_evaluate_fn(test_loader=testloader, model=cfg.model)

    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            fit_config = OmegaConf.to_container(cfg.hyperparams, resolve=True)
            fit_config["curr_round"] = server_round
            cfg.hyperparams.eta_l *= cfg.hyperparams.decay
            return fit_config
        return fit_config_fn

    net_glob = instantiate(cfg.model)

    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(net_glob)),
        net_glob=net_glob,
        epsilon=cfg.hyperparams.epsilon,
        decay=cfg.hyperparams.decay,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
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


if __name__ == '__main__':
    main()
