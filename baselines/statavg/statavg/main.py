"""Run the simulation."""

import os
import pickle
import shutil
from pathlib import Path

import flwr as fl
import hydra
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .client import get_client_fn
from .dataset_preparation import prepare_dataset
from .server import get_evaluate_fn, get_on_fit_config_fn
from .utils import plot_accuracy

# Optional: Force TensorFlow to use the CPU only
# tf.config.set_visible_devices([], 'GPU')

# Optional: Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# If a GPU is used, enable GPU growth for TensorFlow
enable_tf_gpu_growth()


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Define standards for simulation."""
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = Path(HydraConfig.get().runtime.output_dir)

    # 2. Prepare your dataset
    trainset, testset = prepare_dataset(
        cfg.num_clients, cfg.path_to_dataset, cfg.include_testset
    )

    # 3. Define your clients
    client_fn = get_client_fn(
        trainset,
        cfg.scaler_save_path,
        cfg.val_ratio,
        cfg.strategy_name,
        cfg.learning_rate,
    )

    # 4. Define your strategy
    strategy = instantiate(
        cfg.strategy,
        on_fit_config_fn=get_on_fit_config_fn(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(
            testset, cfg.input_shape, cfg.num_classes, cfg.scaler_save_path
        ),
    )

    # 5. Start Simulation

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        ray_init_args={},
        actor_kwargs={"on_actor_init_fn": enable_tf_gpu_growth},
    )

    # 6. Save your results
    results = {"history": history}
    results_path = f"{str(save_path)}/results.pickle"
    with open(results_path, "wb") as file:
        pickle.dump(results, file)

    # Plot averaged accuracy
    plot_accuracy(results_path)

    # (Optional): Delete scalers directory for future experiments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_save_path = os.path.join(script_dir, cfg.scaler_save_path)
    if cfg.delete_scaler_dir:
        if os.path.exists(scaler_save_path):
            shutil.rmtree(scaler_save_path)


if __name__ == "__main__":
    main()
