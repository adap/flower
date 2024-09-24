"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

# These are the basic packages you'll need here

from argparse import Namespace
from typing import Dict

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from mmengine.config import Config
from omegaconf import DictConfig, OmegaConf

from .client import SslClient
from .utils import init_p_paths, load_data, load_model, set_config_mmcv


# pylint: disable=too-many-arguments
def initial_setup(
    cid, base_work_dir, rounds, data_dir, num_gpus, partition_dir, cfg_path
):
    """Initialise setup for instantiating client class."""
    cid_plus_one = str(int(cid) + 1)
    args = Namespace(
        cfg=cfg_path,  # Path to the pretraining configuration file
        checkpoint=None,
        cid=int(cid),
        data_dir=data_dir,
        gpus=num_gpus,
        launcher="none",
        local_rank=0,
        progress=False,
        resume_from=None,
        rounds=rounds,
        seed=7,
        validate=False,
        work_dir=base_work_dir + "/client" + cid_plus_one,
    )

    print("Starting client", args.cid)  # pylint: disable=no-member
    # Fetch the configuration file
    cfg = Config.fromfile(args.cfg)  # pylint: disable=no-member
    # Define the client data files;
    # Usually contains paths to the samples and annotations
    cfg.data.train.data_source.ann_file = (
        partition_dir + "/client_dist" + cid_plus_one + ".json"
    )

    distributed, logger = set_config_mmcv(args, cfg)
    # These two arguments needs to be false during federated pretraining
    # Otherwise the client will load the previously saved checkpoint

    cfg.resume_from = False
    cfg.load_from = False

    # Load the model
    model = load_model(cfg)
    # Load the training data
    train_dataset = load_data(cfg)

    # Since during pretraining of FedVSSL,
    # we don't need any testing data, we can left it empty.
    test_dataset = " "

    return args, cfg, distributed, logger, model, train_dataset, test_dataset


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "epoch_global": str(rnd),
    }
    return config


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Define the main function for FL pre-training or fine-tuning."""
    # Print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    save_path = HydraConfig.get().runtime.output_dir

    # First the paths needs to be defined
    # Otherwise the program may not be able to locate the files of the ctp
    init_p_paths("fedvssl")

    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }
    base_work_dir = save_path + f"/{cfg.exp_name}"
    cfg.strategy.base_work_dir = base_work_dir
    rounds = cfg.rounds
    data_dir = cfg.data_dir + "/ucf101"
    partition_dir = cfg.partition_dir
    num_gpus = int(np.ceil(cfg.client_resources.num_gpus))

    def client_fn(cid: str):
        (
            args,
            config,
            distributed,
            logger,
            model,
            train_dataset,
            test_dataset,
        ) = initial_setup(
            cid,
            base_work_dir,
            rounds,
            data_dir,
            num_gpus,
            partition_dir,
            cfg.cfg_path_pretrain,
        )
        return SslClient(
            model,
            train_dataset,
            test_dataset,
            config,
            args,
            distributed,
            logger,
        )

    # Configure the strategy
    strategy = instantiate(
        cfg.strategy,
        on_fit_config_fn=fit_config,
    )

    # (Optional) Specify ray config
    ray_config = {"include_dashboard": cfg.client_resources.include_dashboard}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
        ray_init_args=ray_config,
    )


if __name__ == "__main__":
    main()
