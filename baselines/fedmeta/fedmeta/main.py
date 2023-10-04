"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from fedmeta.strategy import weighted_average
from dataset import load_datasets
from fedmeta.fedmeta_client_manager import fedmeta_client_manager
import os
import flwr as fl
import client
from utils import save_graph_params, plot_from_pkl


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.

        algo : FedAvg, FedAvg(Meta), FedMeta(MAML), FedMeta(Meta-SGD)
        data : Femnist, Shakespeare

    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, valloaders, testloaders= load_datasets(config=cfg.data, path=cfg.path)

    # prepare function that will be used to spawn each client
    client_fn = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        learning_rate=cfg.algo[cfg.data.data].alpha,
        model=cfg.data.model,
        gradient_step=cfg.data.gradient_step,
    )

    # prepare strategy function
    strategy = instantiate(
        cfg.strategy,
        evaluate_metrics_aggregation_fn=weighted_average,
        alpha=cfg.algo[cfg.data.data].alpha,
        beta=cfg.algo[cfg.data.data].beta,
        data=cfg.data.data,
        algo=cfg.algo.algo,
    )

    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(trainloaders['sup']),
        config=fl.server.ServerConfig(num_rounds=cfg.data.num_rounds),
        client_resources={
            "num_cpus": cfg.data.client_resources.num_cpus,
            "num_gpus": cfg.data.client_resources.num_gpus,
        },
        client_manager=fedmeta_client_manager(valid_client=len(valloaders['qry'])),
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

    print("................")
    print(history)
    output_path = HydraConfig.get().runtime.cwd + '/' + cfg.data.data + '/graph_params'
    os.makedirs(output_path, exist_ok=True)

    data_params = {
        "algo": cfg.algo.algo,
        "data": cfg.data.data,
        "loss": history.losses_distributed,
        "accuracy": history.metrics_distributed,
        "path": output_path
    }

    save_graph_params(data_params)
    plot_from_pkl(directory=f"./{cfg.data.data}/graph_params")
    print("................")


if __name__ == "__main__":
    main()
