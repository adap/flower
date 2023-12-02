"""DepthFL main."""

import copy

import flwr as fl
import hydra
from flwr.common import ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from depthfl import client, server
from depthfl.dataset import load_datasets
from depthfl.utils import save_results_as_pickle


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
    )

    # exclusive learning baseline in DepthFL paper
    # (model_size, % of clients) = (a,100), (b,75), (c,50), (d,25)
    if cfg.exclusive_learning:
        cfg.num_clients = int(
            cfg.num_clients - (cfg.model_size - 1) * (cfg.num_clients // 4)
        )

    models = []
    for i in range(cfg.num_clients):
        model = copy.deepcopy(cfg.model)

        # each client gets different model depth / width
        model.n_blocks = i // (cfg.num_clients // 4) + 1

        # In exclusive learning, every client has same model depth / width
        if cfg.exclusive_learning:
            model.n_blocks = cfg.model_size

        models.append(model)

    # prepare function that will be used to spawn each client
    client_fn = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        learning_rate=cfg.learning_rate,
        learning_rate_decay=cfg.learning_rate_decay,
        models=models,
    )

    # get function that will executed by the strategy's evaluate() method
    # Set server's device
    device = cfg.server_device

    # Static Batch Normalization for HeteroFL
    if cfg.static_bn:
        evaluate_fn = server.gen_evaluate_fn_hetero(
            trainloaders, testloader, device=device, model_cfg=model
        )
    else:
        evaluate_fn = server.gen_evaluate_fn(testloader, device=device, model=model)

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config_fn(server_round):
            # resolve and convert to python dict
            fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
            fit_config["curr_round"] = server_round  # add round info
            return fit_config

        return fit_config_fn

    net = instantiate(cfg.model)
    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    strategy = instantiate(
        cfg.strategy,
        cfg,
        net,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
        initial_parameters=ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in net.state_dict().items()]
        ),
        min_fit_clients=int(cfg.num_clients * cfg.fraction),
        min_available_clients=int(cfg.num_clients * cfg.fraction),
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
        server=server.ServerFedDyn(
            client_manager=SimpleClientManager(), strategy=strategy
        ),
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


if __name__ == "__main__":
    main()
