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
from hfedxgboost.dataset import load_single_dataset

from hfedxgboost.utils import run_centralized
from hfedxgboost.dataset import do_fl_partitioning
from flwr.server.strategy import FedXgbNnAvg
from flwr.server.app import ServerConfig
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import  Parameters, Scalar, EvaluateRes,FitRes,parameters_to_ndarrays
from torch.utils.data import DataLoader, Dataset, TensorDataset
from hfedxgboost.server import serverside_eval,FL_Server
import functools
import flwr as fl
from flwr.common import  Scalar
from hfedxgboost.client import FL_Client
from flwr.server.client_manager import  SimpleClientManager
from hydra.utils import instantiate
from hfedxgboost.utils import results_writer,results_writer_centralized, Early_Stop
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
    if cfg.centralized:
        if cfg.dataset.dataset_name == "all":
            run_centralized(cfg,dataset_name=cfg.dataset.dataset_name)
        if cfg.dataset.dataset_name != "all":
            result_train, result_test=run_centralized(cfg,dataset_name=cfg.dataset.dataset_name)
            writer=results_writer_centralized(cfg)
            #writer.create_res_csv("results_centralized.csv")
            writer.write_res("results_centralized.csv",result_train, result_test)
    else:
        dataset_name=cfg.dataset.dataset_name
        task_type=cfg.dataset.task.task_type
        early_stopper=Early_Stop(cfg)
        X_train,y_train,X_test,y_test=load_single_dataset(task_type,dataset_name,train_ratio=cfg.dataset.train_ratio)
        print("Feature dimension of the dataset:", X_train.shape[1])
        print("Size of the trainset:", X_train.shape[0])
        print("Size of the testset:", X_test.shape[0])
        if task_type=="BINARY":
            print("First class ratio in train data",y_train[y_train==0.0].size/X_train.shape[0])
            print("Second class ratio in train data",y_train[y_train!=.0].size/X_train.shape[0])
            print("First class ratio in test data",y_test[y_test==0.0].size/X_test.shape[0])
            print("Second class ratio in test data",y_test[y_test!=.0].size/X_test.shape[0])
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
            server=FL_Server(client_manager=SimpleClientManager(),early_stopper=early_stopper,
                              strategy=strategy),
            num_clients=client_pool_size,
            client_resources={"num_cpus": cfg.run_experiment.num_cpus_per_client},
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        #print(type(history.metrics_centralized),history.metrics_centralized)
        print(history)
        writer=results_writer(cfg)
        best_res,best_res_round=writer.extract_best_res(history)
        print("Best Result",best_res,"best_res_round",best_res_round)
        #activate the line to create results file from scratch
        #writer.create_res_csv("results.csv")
        writer.write_res("results.csv")
        #return history

if __name__ == "__main__":
    main()
