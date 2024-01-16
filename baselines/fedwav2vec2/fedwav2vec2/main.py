"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedwav2vec2.client import get_client_fn
from fedwav2vec2.models import pre_trained_point
from fedwav2vec2.server import get_evaluate_fn


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    if cfg.pre_train_model_path is not None:
        print("PRETRAINED INITIALIZE")

        pretrained = pre_trained_point(save_path, cfg, cfg.server_device)
    else:
        pretrained = None

    strategy = instantiate(
        cfg.strategy,
        initial_parameters=pretrained,
        evaluate_fn=get_evaluate_fn(
            cfg, server_device=cfg.server_device, save_path=save_path
        ),
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(cfg, save_path),
        num_clients=cfg.total_clients,
        client_resources=cfg.client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
        ray_init_args={"include_dashboard": False},
    )


if __name__ == "__main__":
    main()
