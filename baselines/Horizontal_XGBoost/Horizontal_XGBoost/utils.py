"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
from sklearn.metrics import mean_squared_error, accuracy_score
from hydra.utils import instantiate
from omegaconf import DictConfig

from .dataset import load_single_dataset

from typing import List, Optional, Tuple, Union
from flwr.common import NDArray

from .models import fit_XGBoost,CNN
from torch.utils.data import DataLoader, Dataset, TensorDataset

from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import torch
import torch, torch.nn as nn
from torchmetrics import Accuracy, MeanSquaredError
from tqdm import tqdm

dataset_tasks={
        "a9a":"BINARY",
        "cod-rna":"BINARY",
        "ijcnn1":"BINARY",
        "abalone":"REG",
        "cpusmall":"REG",
        "space_ga":"REG"
    }

def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )
def evaluate(task_type,y,preds):
    if task_type.upper() == "BINARY":
        result = accuracy_score(y, preds)
    elif task_type.upper() == "REG":
        result = mean_squared_error(y, preds)
    return result

#
def run_single_exp(config,dataset_name,task_type,n_estimators):
    X_train,y_train,X_test,y_test=load_single_dataset(task_type,dataset_name,train_ratio=config.dataset.train_ratio)
    tree=fit_XGBoost(config,task_type,X_train,y_train,n_estimators)
    preds_train = tree.predict(X_train)
    result_train=evaluate(task_type,y_train,preds_train)
    preds_test = tree.predict(X_test)
    result_test=evaluate(task_type,y_test,preds_test)
    return result_train,result_test


def run_centralized(config: DictConfig,
                    dataset_name:str ="all",
                    task_type:str =None):
    if dataset_name=="all":
        for dataset in dataset_tasks:
            result_train,result_test=run_single_exp(config,dataset,dataset_tasks[dataset],config.n_estimators)
            print("Results for",dataset,", Task:",dataset_tasks[dataset],", Train:",result_train,", Test:",result_test)
    else:
        if task_type:
            result_train,result_test=run_single_exp(config,dataset,task_type,config.n_estimators)
            print("Results for",dataset,", Task:",task_type,", Train:",result_train,", Test:",result_test)
        else:
            if dataset_name in dataset_tasks.keys():
                result_train,result_test=run_single_exp(config,dataset,dataset_tasks[dataset],config.n_estimators)
                print("Results for",dataset,", Task:",dataset_tasks[dataset],", Train:",result_train,", Test:",result_test)
            else:
                raise Exception(
                    "task_type should be assigned to be BINARY for binary classification tasks" 
                    "or REG for regression tasks"
                    "or the dataset should be one of the follwing"
                    "a9a, cod-rna, ijcnn1, space_ga, abalone, cpusmall"
                    )


def clients_preformance_on_local_data(config: DictConfig,
                                      trainloaders,
                                      X_test,
                                      y_test,
                                      task_type:str):
    n_estimators_client=500//config.client_num
    for i, trainloader in enumerate(trainloaders):
        for local_dataset in trainloader:
            local_X_train, local_y_train = local_dataset[0], local_dataset[1]
            tree=fit_XGBoost(config,task_type,local_X_train, local_y_train,n_estimators_client)#construct_tree(local_X_train, local_y_train, client_tree_num, task_type)

            preds_train = tree.predict(local_X_train)
            result_train=evaluate(task_type,local_y_train,preds_train)

            preds_test = tree.predict(X_test)
            result_test=evaluate(task_type,y_test,preds_test)
            print("Local Client %d XGBoost Training Results: %f" % (i, result_train))
            print("Local Client %d XGBoost Testing Results: %f" % (i, result_test))

#used for both client and server

def single_tree_prediction(
    tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray
) -> Optional[NDArray]:
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return tree.predict(  # type: ignore
        dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )

def single_tree_preds_from_each_client(
            trainloader: DataLoader,
            batch_size: int,
            client_tree_ensemples: Union[
                Tuple[XGBClassifier, int],
                Tuple[XGBRegressor, int],
                List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
            ],
            n_estimators_client: int,
            client_num: int,

    ) -> Optional[Tuple[NDArray, NDArray]]:
        """Extracts each tree from each tree ensemple from each client,
            and predict the output of the data using that tree,
            place those predictions in the preds_from_all_trees_from_all_clients,
            and return it.
            Args:
                trainloader:
                    - a dataloder that contains the dataset to be predicted.
                client_tree_ensemples:
                    - the trained XGBoost tree ensemple from each client, each tree ensemples comes attached
                    to its client id in a tuple
                    - can come as a single tuple of XGBoost tree ensemple and its client id or multiple tuples
                    in one list.
            Returns:

            """
        if trainloader is None:
            return None

        for local_dataset in trainloader:
            x_train, y_train = local_dataset[0], np.float32(local_dataset[1])

        preds_from_all_trees_from_all_clients = np.zeros((x_train.shape[0], client_num * n_estimators_client),dtype=np.float32)

        if isinstance(client_tree_ensemples, list) is False:
            temp_trees = [client_tree_ensemples[0]] * client_num
        elif isinstance(client_tree_ensemples, list):
            client_tree_ensemples.sort(key = lambda x: x[1])
            temp_trees = [i[0] for i in client_tree_ensemples]
            if len(client_tree_ensemples) != client_num:
                temp_trees += ([client_tree_ensemples[0][0]] * (client_num-len(client_tree_ensemples)))

        for i, _ in enumerate(temp_trees):
            for j in range(n_estimators_client):
                preds_from_all_trees_from_all_clients[:, i * n_estimators_client + j] = single_tree_prediction(
                    temp_trees[i], j, x_train
                )

        preds_from_all_trees_from_all_clients = torch.from_numpy(
            np.expand_dims(preds_from_all_trees_from_all_clients, axis=1)
        )
        y_train=torch.from_numpy(
            np.expand_dims(y_train, axis=-1)
        )
        tree_dataset = TensorDataset(preds_from_all_trees_from_all_clients,y_train)
        return get_dataloader(tree_dataset, "tree", batch_size)

from hydra.utils import instantiate

def test(
    cfg,
    #task_type: str,
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True,
) -> Tuple[float, float, int]:

    criterion = instantiate(cfg.dataset.criterion)
    metric_fn= instantiate(cfg.dataset.metric.fn)

    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():

        progress_bar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in progress_bar:
            tree_outputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(tree_outputs)
            total_loss +=criterion(outputs, labels).item()
            n_samples += labels.size(0)
            metric_val = metric_fn(outputs.cpu(), labels.type(torch.int).cpu())
            total_result += metric_val * labels.size(0)

    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples 

#still don't know where to put it
class Early_Stop:
    def __init__(self,task_type, num_waiting_rounds=5):
        self.num_waiting_rounds = num_waiting_rounds
        self.counter = 0
        self.min_loss = float('inf')
        if task_type=="REG":
            self.best_res=float('inf') #mse
            self.compare_fn=min
        if task_type =="BINARY":
            self.best_res=float('inf') #accuracy
            self.compare_fn=max

    def early_stop(self, loss, res):
        if self.compare_fn(res,self.best_res) != self.best_res:
            self.best_res=res
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss):
            self.counter += 1
            if self.counter >= self.num_waiting_rounds:
                return self.best_res
        return None
