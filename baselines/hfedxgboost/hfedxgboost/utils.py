"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
from sklearn.metrics import mean_squared_error, accuracy_score
from hydra.utils import instantiate
from omegaconf import DictConfig

from hfedxgboost.dataset import load_single_dataset

from typing import List, Optional, Tuple, Union
from flwr.common import NDArray

from hfedxgboost.models import fit_XGBoost,CNN
from torch.utils.data import DataLoader, Dataset, TensorDataset

from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import torch
from tqdm import tqdm
from hydra.utils import instantiate
import csv

dataset_tasks={
        "a9a":"BINARY",
        "cod-rna":"BINARY",
        "ijcnn1":"BINARY",
        "abalone":"REG",
        "cpusmall":"REG",
        "space_ga":"REG",
        "YearPredictionMSD":"REG"
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
                    task_type:str =None)-> None:
    if dataset_name=="all":
        for dataset in dataset_tasks:
            result_train,result_test=run_single_exp(config,dataset,dataset_tasks[dataset],config.n_estimators)
            print("Results for",dataset,", Task:",dataset_tasks[dataset],", Train:",result_train,", Test:",result_test)
    else:
        if task_type:
            result_train,result_test=run_single_exp(config,dataset_name,task_type,config.xgboost_params_centralized.n_estimators)
            print("Results for",dataset_name,", Task:",task_type,", Train:",result_train,", Test:",result_test)
            return result_train, result_test
        else:
            if dataset_name in dataset_tasks.keys():
                result_train,result_test=run_single_exp(config,dataset_name,dataset_tasks[dataset_name],config.xgboost_params_centralized.n_estimators)
                print("Results for",dataset_name,", Task:",dataset_tasks[dataset_name],", Train:",result_train,", Test:",result_test)
                return result_train, result_test
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
                                      task_type:str)-> None:
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
    """ 
    This function performs a single tree prediction using the provided tree object on the given dataset. The function accepts the following parameters:

        tree (either XGBClassifier or XGBRegressor): The tree object used for prediction.
        n_tree (int): The index of the tree to be used for prediction.
        dataset (NDArray): The dataset for which the prediction is to be made.

    The function returns an optional NDArray object representing the prediction result. 
    If the provided n_tree is larger than the total number of trees in the tree object, 
    a warning message is printed and None is returned. """

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


def test(
    cfg,
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True,
) -> Tuple[float, float, int]:

    criterion = instantiate(cfg.dataset.task.criterion)
    metric_fn= instantiate(cfg.dataset.task.metric.fn)

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


class Early_Stop:
    def __init__(self,cfg):
        self.num_waiting_rounds = cfg.dataset.early_stop_patience_rounds
        self.counter = 0
        self.min_loss = float('inf')
        self.metric_value=None

    def early_stop(self, res) -> Optional[Tuple[float,float]]:
        """
        check if the model made any progress in <int> number of rounds, if it didn't it will
        return the best result and the server will stop runing the fit function, if it did
        it will return None, and won't stop the server.
        Parameters:
            res: tuple of 2 elements, res[0] is a float that indicate the loss,
            res[1] is actually a 1 element dictionary that looks like this 
            {'Accuracy': tensor(0.8405)}
        Returns:
            Optional[Tuple[float,float]]: (best loss the model achieved,
            best metric value associated with that loss) 
        """
        loss=res[0] 
        metric_val=list(res[1].values())[0].item()
        if loss < self.min_loss:
            self.min_loss = loss
            self.metric_value=metric_val
            self.counter = 0
            print("new best loss value achieved")
        elif loss > (self.min_loss):
            self.counter += 1
            if self.counter >= self.num_waiting_rounds:
                print("That training is been stopped as the model achieve no progress with",
                      "loss =",self.min_loss,
                      "result =",self.metric_value)
                return (self.metric_value,self.min_loss )
        return None

#results 
class results_writer:
    def __init__(self,cfg) -> None:
        self.dataset_name=cfg.dataset.dataset_name
        #self.task_type=cfg.dataset.task_type
        self.n_estimators_client=cfg.n_estimators_client
        self.num_rounds=cfg.run_experiment.num_rounds
        self.client_num=cfg.client_num
        self.xgb_max_depth=cfg.clients.xgb.max_depth
        self.CNN_lr=cfg.clients.CNN.lr
        self.tas_type=cfg.dataset.task.task_type
        self.num_iterations=cfg.run_experiment.fit_config.num_iterations
        if self.tas_type=="REG":
            self.best_res=99999999999.999
            self.compare_fn=min
        if self.tas_type=="BINARY":
            self.best_res=-1
            self.compare_fn=max
        self.best_res_round_num=0
    def extract_best_res(self,history) -> Tuple[float,int]:
        """
        This function takes in a history object and returns the best result and 
        its corresponding round number.
        Parameters:
            history: a history object that contains metrics_centralized keys
        Returns:
            Tuple[float, int]: a tuple containing the best result (float) and 
            its corresponding round number (int)
        """
        for t in history.metrics_centralized.keys():
            l=list()
            print("history.metrics_centralized[t]",history.metrics_centralized[t])
            for i in history.metrics_centralized[t]:
                if self.compare_fn(i[1].item(),self.best_res)==i[1] and i[1].item()!=self.best_res:
                    self.best_res =i[1].item()
                    self.best_res_round_num=i[0]
        return (self.best_res,self.best_res_round_num)
    def create_res_csv(self,filename)-> None:
        fields = ['dataset_name', 'client_num' ,'n_estimators_client',
                   'num_rounds', 'xgb_max_depth', 'CNN_lr',
                   'best_res','best_res_round_num','num_iterations'] 
        with open(filename, 'w') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(fields) 

    def write_res(self,filename) -> None:
        """
        Th function is responsible for writing the results of the 
        federated model to a CSV file. 

        The function opens the specified file in 'a' (append) mode and creates a 
        csvwriter object and add the dataset name, xgboost model's and CNN model's
        hyper-parameters used, and the result.
         
        Parameters: 
            filename: string that indicates the CSV file that will be written in.
        """
        row=[str(self.dataset_name),
             str(self.client_num),
             str(self.n_estimators_client),
             str(self.num_rounds),
             str(self.xgb_max_depth),
             str(self.CNN_lr),
             str(self.best_res),
             str(self.best_res_round_num),
             str(self.num_iterations)]
        with open(filename, 'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)


class results_writer_centralized:
    def __init__(self,cfg) -> None:
        self.dataset_name = cfg.dataset.dataset_name
        self.n_estimators_client = cfg.xgboost_params_centralized.n_estimators
        self.xgb_max_depth = cfg.xgboost_params_centralized.max_depth
        self.tas_type = cfg.dataset.task.task_type
        self.subsample = cfg.xgboost_params_centralized.subsample
        self.learning_rate = cfg.xgboost_params_centralized.learning_rate
        self.colsample_bylevel = cfg.xgboost_params_centralized.colsample_bylevel
        self.colsample_bynode = cfg.xgboost_params_centralized.colsample_bynode
        self.colsample_bytree = cfg.xgboost_params_centralized.colsample_bytree
        self.alpha = cfg.xgboost_params_centralized.alpha
        self.gamma = cfg.xgboost_params_centralized.gamma
        self.num_parallel_tree = cfg.xgboost_params_centralized.num_parallel_tree
        self.min_child_weight = cfg.xgboost_params_centralized.min_child_weight

    def create_res_csv(self,filename)-> None:
        fields = [  "dataset_name",
                    "n_estimators_client",
                    "xgb_max_depth",
                    "subsample",
                    "learning_rate",
                    "colsample_bylevel",
                    "colsample_bynode",
                    "colsample_bytree",
                    "alpha",
                    "gamma",
                    "num_parallel_tree",
                    "min_child_weight",
                    "result_train",
                    "result_test"] 
        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            # writing the fields 
            csvwriter.writerow(fields) 

    def write_res(self,filename, result_train, result_test) -> None:
        """
        Th function is responsible for writing the results of the 
        centralized model to a CSV file. 

        The function opens the specified file in 'a' (append) mode and creates a 
        csvwriter object and add the dataset name, xgboost's 
        hyper-parameters used, and the result.
         
        Parameters: 
            filename: string that indicates the CSV file that will be written in.
        """
        row=[str(self.dataset_name),
            str(self.n_estimators_client),
            str(self.xgb_max_depth),
            str(self.subsample),
            str(self.learning_rate),
            str(self.colsample_bylevel),
            str(self.colsample_bynode),
            str(self.colsample_bytree),
            str(self.alpha),
            str(self.gamma),
            str(self.num_parallel_tree),
            str(self.min_child_weight),
            str(result_train),
            str(result_test)]
        with open(filename, 'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)

    

