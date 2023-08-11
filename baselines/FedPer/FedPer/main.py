"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import flwr as fl
import hydra

from typing import Dict, Any, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from FedPer.utils import utils_file
from FedPer.dataset import load_datasets
from FedPer.strategy import AggregateBodyStrategyPipeline
from FedPer.utils.new_utils import get_client_cls
from hydra.core.hydra_config import HydraConfig
from FedPer.models.cnn_model import CNNNet, CNNModelSplit
from FedPer.utils.base_client import get_fedavg_client_fn
from FedPer.models.mobile_model import MobileNet, MobileNetModelSplit
from FedPer.models.resnet_model import ResNet, ResNetModelSplit
from FedPer.utils.FedPer_client import get_fedper_client_fn

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

    # 2. Prepare your dataset
    trainloader, valloader, testloader = load_datasets(
        config=cfg.dataset,
        num_clients=cfg.num_clients,
    )   

    # 3. Define your clients
    # Get algorithm 
    algorithm = cfg.algorithm.lower()
    # Get client fn 
    if algorithm.lower() == 'fedper':
        client_fn = get_fedper_client_fn(
            trainloaders=trainloader,
            valloaders=valloader,
            model=cfg.model,
        )
    elif algorithm.lower() == 'fedavg':
        client_fn = get_fedavg_client_fn(
            trainloaders=trainloader,
            valloaders=valloader,
            model=cfg.model,
        )
    else: 
        raise NotImplementedError
    
    # Set server's device
    device = cfg.server_device

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
            fit_config["curr_round"] = server_round  # add round info
            return fit_config
        return fit_config_fn
    
    
    if cfg.model.name.lower() == 'cnn':
        split = CNNModelSplit
        def create_model(config: Dict[str, Any]) -> CNNNet:
            """Create initial CNN model."""
            return CNNNet().to(device)
    elif cfg.model.name.lower() == 'mobile':
        split = MobileNetModelSplit
        def create_model() -> MobileNet:
            """Create initial MobileNet-v1 model."""
            return MobileNet(
                num_head_layers=cfg.model.num_head_layers,
                num_classes=cfg.model.num_classes,
                name=cfg.model.name,
                device=cfg.model.device
            ).to(device)
    elif cfg.model.name.lower() == 'resnet':
        split = ResNetModelSplit
        def create_model() -> ResNet:
            """Create initial ResNet model."""
            return ResNet(
                num_head_layers=cfg.model.num_head_layers,
                num_classes=cfg.model.num_classes,
                name=cfg.model.name,
                device=cfg.model.device
            ).to(device)
    else:
        raise NotImplementedError('Model not implemented, check name. ')
        
    # 4. Define your strategy
    strategy = instantiate(
        cfg.strategy,
        create_model=create_model,
        # evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
        model_split_class=split,
    )

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
    utils_file.save_results_as_pickle(history, file_path=save_path, extra_results={})

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

    utils_file.plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )

if __name__ == "__main__":
    main()