"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
from logging import INFO
from math import floor
import os
from typing import Callable, Dict, List, Optional, Tuple
from flwr.common.logger import log
from flwr.common.typing import Metrics
from flwr.server.server import Server
from flwr.server.strategy.fedavg import FedAvg
from power_of_choice.server import PowerOfChoiceCommAndCompVariant
from power_of_choice.models import create_MLP_model, create_CNN_model
from power_of_choice.utils import save_results_as_pickle
from power_of_choice.server import PowerOfChoiceServer
from power_of_choice.client import gen_client_fn
from flwr.server.client_manager import SimpleClientManager
import hydra
import flwr as fl
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


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

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    client_fn = gen_client_fn(cfg.is_cnn)

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

    # Initialize ray_init_args
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }

    # get a function that 

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        initial_learning_rate = 0.005

        def fit_config(server_round: int):
            """Return training configuration dict for each round.

            Take batch size, local epochs and number of samples of each client from the server config
            """

            # learning_rate should be decayed by half every 150 rounds
            exp = server_round // 150
            learning_rate = initial_learning_rate / pow(2, exp)

            config = {
                "batch_size": 32,
                "local_epochs": 1 if server_round < 2 else 2,
                "fraction_samples": None,
                "learning_rate": learning_rate
            }

            config["batch_size"] = cfg.batch_size
            config["local_epochs"] = cfg.local_epochs
            config["fraction_samples"] = cfg.fraction_samples

            print(f"Round {server_round} training config: batch_size={config['batch_size']}, local_epochs={config['local_epochs']}, learning_rate={config['learning_rate']}")

            return config
        
        return fit_config
    
    def get_fit_metrics_aggregation_fn():
        def fit_metrics_aggregation_fn(results: List[Tuple[int, Metrics]]) -> Metrics:
            # Initialize lists to store training losses
            training_losses = []

            # Extract training losses and client counts from results
            for _, metrics in results:
                if 'training_loss' in metrics:
                    training_loss = metrics['training_loss']
                    training_losses.append(training_loss)

            # Calculate the variance and average of training loss
            variance_training_loss = np.var(training_losses)
            average_training_loss = np.mean(training_losses)

            # Create the aggregated metrics dictionary
            aggregated_metrics = {
                'variance_training_loss': variance_training_loss,
                'average_training_loss': average_training_loss
            }

            return aggregated_metrics
        
        return fit_metrics_aggregation_fn
    
    def get_on_evaluate_config(is_cpow: bool, b: Optional[int] = None):
        def evaluate_config(server_round: int):
            """Return evaluation configuration dict for each round.

            In case we are using cpow variant, we set b to the value specified in the configuration file.
            """

            config = {
                "is_cpow": False,
            }

            if is_cpow:
                config["is_cpow"] = True
                config["b"] = b

            return config
        
        return evaluate_config
    
    def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""
        
        print(f"Current folder is {os.getcwd()}")

        test_folder = "fmnist"
        if cfg.is_cnn:
            test_folder = "cifar10"

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        x_test = np.load(os.path.join(test_folder, "x_test.npy"))
        y_test = np.load(os.path.join(test_folder, "y_test.npy"))

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
            return loss, {"accuracy": accuracy}

        return evaluate
    
    if cfg.is_cnn:
        server_model = create_CNN_model()
    else:
        server_model = create_MLP_model()
    
    server_model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    is_cpow = False
    is_rpow = False
    is_rand = False
    if cfg.variant == "cpow":
        is_cpow = True
    elif cfg.variant == "rpow":
        is_rpow = True
    elif cfg.variant == "rand":
        is_rand = True

    
    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    if is_rpow:
        # Build Atmp dictionary with number of clients items with key = client id and value = inf
        atmp = {}
        for i in range(cfg.num_clients):
            atmp[str(i)] = float("inf")
        
        # Instantiate strategy
        strategy = instantiate(
            cfg.strategy,
            variant="rpow",
            atmp=atmp,
            on_fit_config_fn=get_on_fit_config(),
            evaluate_fn=get_evaluate_fn(server_model),
            on_evaluate_config_fn=get_on_evaluate_config(is_cpow),
        )
    elif is_rand:
        # Instantiate FedAvg strategy
        strategy = FedAvg(
            fraction_fit=round(cfg.strategy.ck / cfg.num_clients, 2),
            fraction_evaluate=round(cfg.strategy.ck / cfg.num_clients, 2),
            min_fit_clients=round(cfg.strategy.ck / cfg.num_clients),
            min_evaluate_clients=round(cfg.strategy.ck / cfg.num_clients),
            on_fit_config_fn=get_on_fit_config(),
            evaluate_fn=get_evaluate_fn(server_model),
            on_evaluate_config_fn=get_on_evaluate_config(is_cpow, cfg.b),
            fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn()
        )
    else:
        # Instantiate strategy with base config
        strategy = instantiate(
            cfg.strategy,
            on_fit_config_fn=get_on_fit_config(),
            evaluate_fn=get_evaluate_fn(server_model),
            on_evaluate_config_fn=get_on_evaluate_config(is_cpow, cfg.b),
            fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn()
        )

    client_manager = SimpleClientManager()

    if is_rpow:
        # Instantiate rpow server with strategy and client manager
        server = PowerOfChoiceCommAndCompVariant(strategy=strategy, client_manager=client_manager)
    elif is_rand:
        log(INFO, "Using FedAvg strategy")
        server = Server(strategy=strategy, client_manager=client_manager)
    else:
        # Instantiate base server with strategy and client manager
        server = PowerOfChoiceServer(strategy=strategy, client_manager=client_manager)

    # 5. Start Simulation

    print("Starting simulation")

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        server = server,
        ray_init_args=ray_init_args,
    )

    # 6. Save your results
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

    # # plot results and include them in the readme
    # strategy_name = strategy.__class__.__name__
    # file_suffix: str = (
    #     f"_{strategy_name}"
    #     f"_C={cfg.num_clients}"
    #     f"_B={cfg.batch_size}"
    #     f"_E={cfg.local_epochs}"
    #     f"_R={cfg.num_rounds}"
    #     f"_d={cfg.strategy.d}"
    #     f"_CK={cfg.strategy.ck}"
    # )

    # plot_metric_from_history(
    #     history,
    #     save_path,
    #     (file_suffix),
    # )
if __name__ == "__main__":
    main()