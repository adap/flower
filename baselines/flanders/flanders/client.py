"""Clients implementation for Flanders."""
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple, Union

import flwr as fl
import numpy as np
import ray
import torch
from flwr.common.typing import Scalar
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from .dataset import get_dataloader, get_mnist, cifar10_transformation, mnist_transformation
from .models import CifarNet, MnistNet, test_cifar, test_mnist, train_cifar, train_mnist

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]


def get_params(model):
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, params):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_sklearn_model_params(model):
    """Return the paramters of a sklearn LogisticRegression model."""
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


def set_sklearn_model_params(model, params):
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params_logistic_regr(model):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called.

    Server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 2
    n_features = 14
    model.classes_ = np.array(list(range(n_classes)))

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
    return model


def set_initial_params_linear_regr(model):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called.

    Server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LinearRegression documentation for more information.
    """
    n_features = 18
    model.coef_ = np.zeros((n_features,))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))
    return model


# Adapted from Pytorch quickstart example
class CifarClient(fl.client.NumPyClient):
    """Implementation of CIFAR-10 image classification using PyTorch."""

    def __init__(self, cid, fed_dir_data):
        """Instantiate a client for the CIFAR-10 dataset."""
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = CifarNet()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps")

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy ndarrays."""
        return get_params(self.net)

    def fit(self, parameters, config):
        """Set model parameters from a list of NumPy ndarrays."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 1
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            transform=cifar10_transformation,
        )

        self.net.to(self.device)
        train_cifar(self.net, trainloader, epochs=config["epochs"], device=self.device)

        return (
            get_params(self.net),
            len(trainloader.dataset),
            {"cid": self.cid},
        )

    def evaluate(self, parameters, config):
        """Evaluate using local test dataset."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers, transform=cifar10_transformation
        )

        self.net.to(self.device)
        loss, accuracy = test_cifar(self.net, valloader, device=self.device)

        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

class MnistClient(fl.client.NumPyClient):
    """Implementation of MNIST image classification using PyTorch."""

    def __init__(self, cid, fed_dir_data):
        """Instantiate a client for the MNIST dataset."""
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = MnistNet()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps")

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy ndarrays."""
        return get_params(self.net)

    def fit(self, parameters, config):
        """Set model parameters from a list of NumPy ndarrays."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 1
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            transform=mnist_transformation,
        )

        self.net.to(self.device)
        train_mnist(self.net, trainloader, epochs=config["epochs"], device=self.device)

        return (
            get_params(self.net),
            len(trainloader.dataset),
            {"cid": self.cid},
        )

    def evaluate(self, parameters, config):
        """Evaluate using local test dataset."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers, transform=mnist_transformation
        )

        self.net.to(self.device)
        loss, accuracy = test_mnist(self.net, valloader, device=self.device)

        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

class IncomeClient(fl.client.NumPyClient):
    """Implementation income classification using scikit-learn."""

    # pylint: disable=too-many-arguments
    def __init__(self, cid: str, x_train, y_train, x_test, y_test):
        """Instantiate a client for the income dataset."""
        self.model = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=1, warm_start=True
        )
        set_initial_params_logistic_regr(self.model)
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config: Dict[str, Scalar]):
        """Get model parameters as a list of NumPy ndarrays."""
        return get_sklearn_model_params(self.model)

    def fit(self, parameters, config):
        """Set model parameters from a list of NumPy ndarrays."""
        # Set scikit logistic regression model parameters
        self.model = set_sklearn_model_params(self.model, parameters)
        self.model.fit(self.x_train, self.y_train)
        new_parameters = get_sklearn_model_params(self.model)
        return (
            new_parameters,
            len(self.x_train),
            {"cid": self.cid},
        )

    def evaluate(self, parameters, config):
        """Evaluate using local test dataset."""
        # Set scikit logistic regression model parameters
        self.model = set_sklearn_model_params(self.model, parameters)
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(y_pred, self.y_test)
        loss = log_loss(self.y_test, self.model.predict_proba(self.x_test))
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


class HouseClient(fl.client.NumPyClient):
    """Implementation of house price prediction using scikit-learn."""

    # pylint: disable=too-many-arguments
    def __init__(self, cid: str, x_train, y_train, x_test, y_test):
        """Instantiate a client for the house dataset."""
        self.model = ElasticNet(alpha=1, max_iter=1, warm_start=True)
        set_initial_params_linear_regr(self.model)
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config: Dict[str, Scalar]):
        """Get model parameters as a list of NumPy ndarrays."""
        return get_sklearn_model_params(self.model)

    def fit(self, parameters, config):
        """Set model parameters from a list of NumPy ndarrays."""
        self.model = set_sklearn_model_params(self.model, parameters)
        self.model.fit(self.x_train, self.y_train)
        new_parameters = get_sklearn_model_params(self.model)
        return (
            new_parameters,
            len(self.x_train),
            {"cid": self.cid},
        )

    def evaluate(self, parameters, config):
        """Evaluate using local test dataset."""
        raise NotImplementedError("HouseClient.evaluate not implemented")
