"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Dict, List, Union, Tuple
from pathlib import Path
from neural_networks.dataset_utils import get_mnist, do_fl_partitioning, get_dataloader, get_circles
from neural_networks.neural_networks import CifarNet, train_cifar, test_cifar, MnistNet, ToyNN, test_toy, train_mnist, test_mnist, train_toy
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, log_loss

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_sklearn_model_params(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_sklearn_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params_logistic_regr(model: LogisticRegression):
    """
    Sets initial parameters as zeros. Required since model params are
    uninitialized until model.fit is called.

    Server asks for initial parameters from clients at launch. 
    Refer to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2
    n_features = 14
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
    return model

def set_initial_params_linear_regr(model: ElasticNet):
    """
    Sets initial parameters as zeros. Required since model params are
    uninitialized until model.fit is called.

    Server asks for initial parameters from clients at launch. 
    Refer to sklearn.linear_model.LinearRegression documentation for more
    information.
    """
    n_features = 18
    model.coef_ = np.zeros((n_features,))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))
    return model


# Adapted from Pytorch quickstart example
class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid: str, pool_size: int):
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = MnistNet()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pool_size = pool_size

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 1

        trainloader = get_mnist("datasets", 32, self.cid, nb_clients=self.pool_size, is_train=True, workers=num_workers)

        # Send model to device
        self.net.to(self.device)

        # Train
        train_mnist(self.net, trainloader, epochs=config["epochs"], device=self.device)

        new_parameters = self.get_parameters(config={})

        # Return local model and statistics
        return new_parameters, len(trainloader.dataset), {"malicious": config["malicious"], "cid": self.cid}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        testloader = get_mnist("datasets", 32, self.cid, nb_clients=self.pool_size, is_train=False, workers=num_workers)

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test_mnist(self.net, testloader, device=self.device)

        # Return statistics
        return float(loss), len(testloader), {"accuracy": float(accuracy)}


# Adapted from Pytorch quickstart example
class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = CifarNet()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("mps")
    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 1
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )
        
        self.net.to(self.device)
        train_cifar(self.net, trainloader, epochs=config["epochs"], device=self.device)
        
        return get_params(self.net), len(trainloader.dataset), {"malicious": config["malicious"], "cid": self.cid}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        self.net.to(self.device)
        loss, accuracy = test_cifar(self.net, valloader, device=self.device)

        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


class IncomeClient(fl.client.NumPyClient):
    def __init__(self, cid: str, x_train, y_train, x_test, y_test):
        self.model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1, warm_start=True)
        set_initial_params_logistic_regr(self.model)
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_sklearn_model_params(self.model)

    def fit(self, parameters, config):
        # Set scikit logistic regression model parameters
        self.model = set_sklearn_model_params(self.model, parameters)
        self.model.fit(self.x_train, self.y_train)
        new_parameters = get_sklearn_model_params(self.model)
        return new_parameters, len(self.x_train), {"malicious": config["malicious"], "cid": self.cid}
    
    def evaluate(self, parameters, config):
        # Set scikit logistic regression model parameters
        self.model = set_sklearn_model_params(self.model, parameters)
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(y_pred, self.y_test)
        loss = log_loss(self.y_test, self.model.predict_proba(self.x_test))
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


class HouseClient(fl.client.NumPyClient):
    def __init__(self, cid: str, x_train, y_train, x_test, y_test):
        self.model = ElasticNet(alpha=1, max_iter=1, warm_start=True)
        set_initial_params_linear_regr(self.model)
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_sklearn_model_params(self.model)

    def fit(self, parameters, config):
        self.model = set_sklearn_model_params(self.model, parameters)
        self.model.fit(self.x_train, self.y_train)
        new_parameters = get_sklearn_model_params(self.model)
        return new_parameters, len(self.x_train), {"malicious": config["malicious"], "cid": self.cid}
    
    def evaluate(self, parameters, config):
        return 0, 0, {"accuracy": 0}


class ToyClient(fl.client.NumPyClient):
    def __init__(self, cid: str, pool_size: int):
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = ToyNN()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        #num_workers = len(ray.worker.get_resource_ids()["CPU"])
        num_workers = 1
        trainloader = get_circles(32, n_samples=10000, workers=num_workers, is_train=True)

        self.net.to(self.device)
        train_toy(self.net, trainloader, epochs=config["epochs"], device=self.device)

        new_parameters = self.get_parameters(config={})

        # Return local model and statistics
        return new_parameters, len(trainloader.dataset), {"malicious": config["malicious"], "cid": self.cid}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        #num_workers = len(ray.worker.get_resource_ids()["CPU"])
        num_workers = 1
        testloader = get_circles(32, n_samples=10000, workers=num_workers, is_train=False)

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test_toy(self.net, testloader, device=self.device)

        # Return statistics
        return float(loss), len(testloader), {"accuracy": float(accuracy)}