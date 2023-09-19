"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset
import torch
from .dataset import load_single_dataset

from .utils import run_centralized
from .dataset import do_fl_partitioning
from flwr.server.strategy import FedXgbNnAvg
from flwr.server.app import ServerConfig
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import  Parameters, Scalar, EvaluateRes,FitRes,parameters_to_ndarrays
from torch.utils.data import DataLoader, Dataset, TensorDataset
from xgboost import XGBClassifier, XGBRegressor
from .server import serverside_eval,FL_Server
import functools
import flwr as fl
from flwr.common import  Parameters, Scalar
from .client import FL_Client
from flwr.server.client_manager import  SimpleClientManager
from hydra.utils import instantiate

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    dataset_tasks={
        "a9a":"BINARY",
        "cod-rna":"BINARY",
        "ijcnn1":"BINARY",
        "abalone":"REG",
        "cpusmall":"REG",
        "space_ga":"REG"
    }
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))
    if cfg.centralized:
        run_centralized(cfg,dataset_name=cfg.dataset.dataset_name)
    else:
        dataset_name=cfg.dataset.dataset_name
        task_type=cfg.dataset.task_type
        X_train,y_train,X_test,y_test=load_single_dataset(task_type,dataset_name,train_ratio=cfg.dataset.train_ratio)
        trainset=TensorDataset(torch.from_numpy(X_train), torch.from_numpy (y_train))
        testset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy (y_test))

        trainloaders, valloaders, testloader = do_fl_partitioning(
                                                                trainset,
                                                                testset,
                                                                batch_size=cfg.batch_size,
                                                                pool_size=cfg.client_num,
                                                                val_ratio=cfg.val_ratio,)
        print(
        f"Data partitioned across {cfg.client_num} clients"
        f" and {cfg.val_ratio} of local dataset reserved for validation."
        )
        #clients_preformance_on_local_data(cfg,trainloaders,X_test,y_test,task_type)
        """   run_experiment(
            cfg=cfg,
            trainloaders=trainloaders,  
            valloaders=valloaders,
            testloader=testloader,
        )"""
        num_rounds=cfg.run_experiment.num_rounds
        client_pool_size=cfg.client_num
        batch_size=cfg.run_experiment.batch_size
        val_ratio=cfg.val_ratio 

        # Configure the strategy
        def fit_config(server_round: int) -> Dict[str, Scalar]:
            print(f"Configuring round {server_round}")
            return {
                "num_iterations": cfg.run_experiment.fit_config.num_iterations,
                "batch_size": batch_size,
            }

        # FedXgbNnAvg
        strategy =instantiate(cfg.strategy,
                              on_fit_config_fn=fit_config,
                              on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
                              evaluate_fn=functools.partial(serverside_eval,
                                                            cfg=cfg,
                                                            testloader=testloader,
                                                            batch_size=batch_size,)
                              )

        print(
            f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool."
        )

        def client_fn(cid: str) -> fl.client.Client:
            """Creates a federated learning client"""
            if val_ratio > 0.0 and val_ratio <= 1.0:
                return FL_Client(
                    cfg,
                    trainloaders[int(cid)],
                    valloaders[int(cid)],
                    client_pool_size,
                    cid,
                    log_progress=False,
                )
            else:
                return FL_Client(
                    cfg,
                    trainloaders[int(cid)],
                    None,
                    client_pool_size,
                    cid,
                    log_progress=False,
                )

        # Start the simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            server=FL_Server(client_manager=SimpleClientManager(), strategy=strategy),
            num_clients=client_pool_size,
            client_resources={"num_cpus": cfg.run_experiment.num_cpus_per_client},
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        print(history)

        #return history

if __name__ == "__main__":
    main()
