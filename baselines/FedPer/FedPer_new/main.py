"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import flwr as fl
import hydra

from typing import Dict, Any, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from FedPer_new.dataset import dataset_main
from hydra.core.hydra_config import HydraConfig
from FedPer_new.models.cnn_model import CNNNet, CNNModelSplit
from FedPer_new.models.mobile_model import MobileNet, MobileNetModelSplit
from FedPer_new.models.resnet_model import ResNet, ResNetModelSplit

from FedPer_new.fedavg_client import gen_client_fn
from FedPer_new.utils import weighted_average, save_results_as_pickle, plot_metric_from_history
from flwr.server.strategy import FedAvg

@hydra.main(config_path="conf", config_name="new_base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """  

    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))
    if cfg.model.name.lower() == 'resnet':
        cfg.model._target_ = 'FedPer_new.model.ResNet18'
    elif cfg.model.name.lower() == 'mobile':
        cfg.model._target_ = 'FedPer_new.model.MobileNet_v2'
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented")

    # 2. Prepare your dataset
    dataset_main(cfg.dataset)
    
    # 3. Define your clients
    # Get algorithm 
    algo = cfg.algo.lower()

    # Get client fn 
    if algo == 'fedavg':
        client_fn = gen_client_fn(cfg)
    else: 
        raise NotImplementedError(f"Algorithm {algo} not implemented")

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
            fit_config["curr_round"] = server_round  # add round info
            return fit_config
        return fit_config_fn
        
    # 4. Define your strategy
    if algo == 'fedavg':
        strategy = FedAvg(
            fraction_fit=cfg.strategy.fraction_fit,
            fraction_evaluate=cfg.strategy.fraction_evaluate,
            min_fit_clients=cfg.strategy.min_fit_clients,
            min_evaluate_clients=cfg.strategy.min_evaluate_clients,
            min_available_clients=cfg.strategy.min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        raise NotImplementedError(f"Algorithm {algo} not implemented")

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_lr={cfg.learning_rate}"
    )

    plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )

if __name__ == "__main__":
    main()