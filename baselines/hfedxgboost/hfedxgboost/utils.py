"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import csv
import math
import os
import os.path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import NDArray
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

from hfedxgboost.dataset import load_single_dataset
from hfedxgboost.models import CNN, fit_xgboost

dataset_tasks = {
    "a9a": "BINARY",
    "cod-rna": "BINARY",
    "ijcnn1": "BINARY",
    "abalone": "REG",
    "cpusmall": "REG",
    "space_ga": "REG",
    "YearPredictionMSD": "REG",
}


def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    """Create a DataLoader object for the given dataset.

    Parameters
    ----------
     dataset (Dataset): The dataset object containing the data to be loaded.
     partition (str): The partition of the dataset to load.
     batch_size (Union[int, str]): The size of each mini-batch.
    If "whole" is specified, the entire dataset will be included in a single batch.

    Returns
    -------
     loader (DataLoader): The DataLoader object.
    """
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


def evaluate(task_type, y, preds) -> float:
    """Evaluate the performance of a given model prediction based on the task type.

    Parameters
    ----------
        task_type: A string representing the type of the task
        (either "BINARY" or "REG").
        y: The true target values.
        preds: The predicted target values.

    Returns
    -------
        result: The evaluation result based on the task type. If task_type is "BINARY",
        it computes the accuracy score.
        If task_type is "REG", it calculates the mean squared error.
    """
    if task_type.upper() == "BINARY":
        result = accuracy_score(y, preds)
    elif task_type.upper() == "REG":
        result = mean_squared_error(y, preds)
    return result


def run_single_exp(
    config, dataset_name, task_type, n_estimators
) -> Tuple[float, float]:
    """Run a single experiment using XGBoost on a dataset.

    Parameters
    ----------
    - config (object): Hydra Configuration object containing necessary settings.
    - dataset_name (str): Name of the dataset to train and test on.
    - task_type (str): Type of the task "BINARY" or "REG".
    - n_estimators (int): Number of estimators (trees) to use in the XGBoost model.

    Returns
    -------
    - result_train (float): Evaluation result on the training set.
    - result_test (float): Evaluation result on the test set.
    """
    x_train, y_train, x_test, y_test = load_single_dataset(
        task_type, dataset_name, train_ratio=config.dataset.train_ratio
    )
    tree = fit_xgboost(config, task_type, x_train, y_train, n_estimators)
    preds_train = tree.predict(x_train)
    result_train = evaluate(task_type, y_train, preds_train)
    preds_test = tree.predict(x_test)
    result_test = evaluate(task_type, y_test, preds_test)
    return result_train, result_test


def run_centralized(
    config: DictConfig, dataset_name: str = "all", task_type: Optional[str] = None
) -> Union[Tuple[float, float], List[None]]:
    """Run the centralized training and testing process.

    Parameters
    ----------
        config (DictConfig): Hydra configuration object.
        dataset_name (str): Name of the dataset to run the experiment on.
        task_type (str): Type of task.

    Returns
    -------
        None: Returns None if dataset_name is "all".
        Tuple: Returns a tuple (result_train, result_test) of training and testing
        results if dataset_name is not "all" and task_type is specified.

    Raises
    ------
        Exception: Raises an exception if task_type is not specified correctly
        and the dataset_name is not in the dataset_tasks dict.
    """
    if dataset_name == "all":
        for dataset in dataset_tasks:
            result_train, result_test = run_single_exp(
                config, dataset, dataset_tasks[dataset], config.n_estimators
            )
            print(
                "Results for",
                dataset,
                ", Task:",
                dataset_tasks[dataset],
                ", Train:",
                result_train,
                ", Test:",
                result_test,
            )
        return []

    if task_type:
        result_train, result_test = run_single_exp(
            config,
            dataset_name,
            task_type,
            config.xgboost_params_centralized.n_estimators,
        )
        print(
            "Results for",
            dataset_name,
            ", Task:",
            task_type,
            ", Train:",
            result_train,
            ", Test:",
            result_test,
        )
        return result_train, result_test

    if dataset_name in dataset_tasks.keys():
        result_train, result_test = run_single_exp(
            config,
            dataset_name,
            dataset_tasks[dataset_name],
            config.xgboost_params_centralized.n_estimators,
        )
        print(
            "Results for",
            dataset_name,
            ", Task:",
            dataset_tasks[dataset_name],
            ", Train:",
            result_train,
            ", Test:",
            result_test,
        )
        return result_train, result_test

    raise Exception(
        "task_type should be assigned to be BINARY for"
        "binary classification"
        "tasks or REG for regression tasks"
        "or the dataset should be one of the follwing"
        "a9a, cod-rna, ijcnn1, space_ga, abalone,",
        "cpusmall, YearPredictionMSD",
    )


def local_clients_performance(
    config: DictConfig, trainloaders, x_test, y_test, task_type: str
) -> None:
    """Evaluate the performance of clients on local data using XGBoost.

    Parameters
    ----------
        config (DictConfig): Hydra configuration object.
        trainloaders: List of data loaders for each client.
        x_test: Test features.
        y_test: Test labels.
        task_type (str): Type of prediction task.
    """
    for i, trainloader in enumerate(trainloaders):
        for local_dataset in trainloader:
            local_x_train, local_y_train = local_dataset[0], local_dataset[1]
            tree = fit_xgboost(
                config,
                task_type,
                local_x_train,
                local_y_train,
                500 // config.client_num,
            )

            preds_train = tree.predict(local_x_train)
            result_train = evaluate(task_type, local_y_train, preds_train)

            preds_test = tree.predict(x_test)
            result_test = evaluate(task_type, y_test, preds_test)
            print("Local Client %d XGBoost Training Results: %f" % (i, result_train))
            print("Local Client %d XGBoost Testing Results: %f" % (i, result_test))


def single_tree_prediction(
    tree,
    n_tree: int,
    dataset: NDArray,
) -> Optional[NDArray]:
    """Perform a single tree prediction using the provided tree object on given dataset.

    Parameters
    ----------
        tree (either XGBClassifier or XGBRegressor): The tree object
        used for prediction.
        n_tree (int): The index of the tree to be used for prediction.
        dataset (NDArray): The dataset for which the prediction is to be made.

    Returns
    -------
        NDArray object: representing the prediction result.
        None: If the provided n_tree is larger than the total number of trees
        in the tree object, and a warning message is printed.
    """
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return tree.predict(
        dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )


def single_tree_preds_from_each_client(
    trainloader: DataLoader,
    batch_size,
    client_tree_ensamples: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    n_estimators_client: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Predict using trees from client tree ensamples.

    Extract each tree from each tree ensample from each client,
    and predict the output of the data using that tree,
    place those predictions in the preds_from_all_trees_from_all_clients,
    and return it.

    Parameters
    ----------
        trainloader:
            - a dataloader that contains the dataset to be predicted.
        client_tree_ensamples:
            - the trained XGBoost tree ensample from each client,
            each tree ensembles comes attached
            to its client id in a tuple
            - can come as a single tuple of XGBoost tree ensample and
            its client id or multiple tuples in one list.

    Returns
    -------
        loader (DataLoader): The DataLoader object that contains the
        predictions of the tree
    """
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], np.float32(local_dataset[1])

    preds_from_all_trees_from_all_clients = np.zeros(
        (x_train.shape[0], client_num * n_estimators_client), dtype=np.float32
    )
    if isinstance(client_tree_ensamples, list) is False:
        temp_trees = [client_tree_ensamples[0]] * client_num
    elif isinstance(client_tree_ensamples, list):
        client_tree_ensamples.sort(key=lambda x: x[1])
        temp_trees = [i[0] for i in client_tree_ensamples]
        if len(client_tree_ensamples) != client_num:
            temp_trees += [client_tree_ensamples[0][0]] * (
                client_num - len(client_tree_ensamples)
            )

    for i, _ in enumerate(temp_trees):
        for j in range(n_estimators_client):
            preds_from_all_trees_from_all_clients[:, i * n_estimators_client + j] = (
                single_tree_prediction(temp_trees[i], j, x_train)
            )

    preds_from_all_trees_from_all_clients = torch.from_numpy(
        np.expand_dims(preds_from_all_trees_from_all_clients, axis=1)
    )
    y_train = torch.from_numpy(np.expand_dims(y_train, axis=-1))
    tree_dataset = TensorDataset(preds_from_all_trees_from_all_clients, y_train)
    return get_dataloader(tree_dataset, "tree", batch_size)


def test(
    cfg,
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    """Evaluates the performance of a CNN model on a given test dataset.

    Parameters
    ----------
        cfg (Any): The configuration object.
        net (CNN): The CNN model to test.
        testloader (DataLoader): The data loader for the test dataset.
        device (torch.device): The device to run the evaluation on.
        log_progress (bool, optional): Whether to log the progress during evaluation.
        Default is True.

    Returns
    -------
        Tuple[float, float, int]: A tuple containing the average loss,
        average metric result, and total number of samples evaluated.
    """
    criterion = instantiate(cfg.dataset.task.criterion)
    metric_fn = instantiate(cfg.dataset.task.metric.fn)

    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():
        # progress_bar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in tqdm(testloader, desc="TEST") if log_progress else testloader:
            tree_outputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(tree_outputs)
            total_loss += criterion(outputs, labels).item()
            n_samples += labels.size(0)
            metric_val = metric_fn(outputs.cpu(), labels.type(torch.int).cpu())
            total_result += metric_val * labels.size(0)

    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples


class EarlyStop:
    """Stop the tain when no progress is happening."""

    def __init__(self, cfg):
        self.num_waiting_rounds = cfg.dataset.early_stop_patience_rounds
        self.counter = 0
        self.min_loss = float("inf")
        self.metric_value = None

    def early_stop(self, res) -> Optional[Tuple[float, float]]:
        """Check if the model made any progress in number of rounds.

        If it didn't it will return the best result and the server
        will stop running the fit function, if
        it did it will return None, and won't stop the server.

        Parameters
        ----------
            res: tuple of 2 elements, res[0] is a float that indicate the loss,
            res[1] is actually a 1 element dictionary that looks like this
            {'Accuracy': tensor(0.8405)}

        Returns
        -------
            Optional[Tuple[float,float]]: (best loss the model achieved,
            best metric value associated with that loss)
        """
        loss = res[0]
        metric_val = list(res[1].values())[0].item()
        if loss < self.min_loss:
            self.min_loss = loss
            self.metric_value = metric_val
            self.counter = 0
            print(
                "New best loss value achieved,",
                "loss",
                self.min_loss,
                "metric value",
                self.metric_value,
            )
        elif loss > (self.min_loss):
            self.counter += 1
            if self.counter >= self.num_waiting_rounds:
                print(
                    "That training is been stopped as the",
                    "model achieve no progress with",
                    "loss =",
                    self.min_loss,
                    "result =",
                    self.metric_value,
                )
                return (self.metric_value, self.min_loss)
        return None


# results


def create_res_csv(filename, fields) -> None:
    """Create a CSV file with the provided file name."""
    if not os.path.isfile(filename):
        with open(filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)


class ResultsWriter:
    """Write the results for the federated experiments."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tas_type = cfg.dataset.task.task_type
        if self.tas_type == "REG":
            self.best_res = math.inf
            self.compare_fn = min
        if self.tas_type == "BINARY":
            self.best_res = -1
            self.compare_fn = max
        self.best_res_round_num = 0
        self.fields = [
            "dataset_name",
            "client_num",
            "n_estimators_client",
            "num_rounds",
            "xgb_max_depth",
            "cnn_lr",
            "best_res",
            "best_res_round_num",
            "num_iterations",
        ]

    def extract_best_res(self, history) -> Tuple[float, int]:
        """Take the history & returns the best result and its corresponding round num.

        Parameters
        ----------
            history: a history object that contains metrics_centralized keys

        Returns
        -------
            Tuple[float, int]: a tuple containing the best result (float) and
            its corresponding round number (int)
        """
        for key in history.metrics_centralized.keys():
            for i in history.metrics_centralized[key]:
                if (
                    self.compare_fn(i[1].item(), self.best_res) == i[1]
                    and i[1].item() != self.best_res
                ):
                    self.best_res = i[1].item()
                    self.best_res_round_num = i[0]
        return (self.best_res, self.best_res_round_num)

    def write_res(self, filename) -> None:
        """Write the results of the federated model to a CSV file.

        The function opens the specified file in 'a' (append) mode and creates a
        csvwriter object and add the dataset name, xgboost model's and CNN model's
        hyper-parameters used, and the result.

        Parameters
        ----------
            filename: string that indicates the CSV file that will be written in.
        """
        row = [
            str(self.cfg.dataset.dataset_name),
            str(self.cfg.client_num),
            str(self.cfg.clients.n_estimators_client),
            str(self.cfg.run_experiment.num_rounds),
            str(self.cfg.clients.xgb.max_depth),
            str(self.cfg.clients.CNN.lr),
            str(self.best_res),
            str(self.best_res_round_num),
            str(self.cfg.run_experiment.fit_config.num_iterations),
        ]
        with open(filename, "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)


class CentralizedResultsWriter:
    """Write the results for the centralized experiments."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tas_type = cfg.dataset.task.task_type
        self.fields = [
            "dataset_name",
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
            "result_test",
        ]

    def write_res(self, filename, result_train, result_test) -> None:
        """Write the results of the centralized model to a CSV file.

        The function opens the specified file in 'a' (append) mode and creates a
        csvwriter object and add the dataset name, xgboost's
        hyper-parameters used, and the result.

        Parameters
        ----------
            filename: string that indicates the CSV file that will be written in.
        """
        row = [
            str(self.cfg.dataset.dataset_name),
            str(self.cfg.xgboost_params_centralized.n_estimators),
            str(self.cfg.xgboost_params_centralized.max_depth),
            str(self.cfg.xgboost_params_centralized.subsample),
            str(self.cfg.xgboost_params_centralized.learning_rate),
            str(self.cfg.xgboost_params_centralized.colsample_bylevel),
            str(self.cfg.xgboost_params_centralized.colsample_bynode),
            str(self.cfg.xgboost_params_centralized.colsample_bytree),
            str(self.cfg.xgboost_params_centralized.alpha),
            str(self.cfg.xgboost_params_centralized.gamma),
            str(self.cfg.xgboost_params_centralized.num_parallel_tree),
            str(self.cfg.xgboost_params_centralized.min_child_weight),
            str(result_train),
            str(result_test),
        ]
        with open(filename, "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)
