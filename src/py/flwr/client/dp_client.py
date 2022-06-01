"""Differentially Private Client."""
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from opacus import PrivacyEngine
from torch import Generator
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from flwr.client.numpy_client import NumPyClient
from flwr.common import Scalar


class DPClient(NumPyClient):
    """Differentially private version of NumPyClient using Opacus and PyTorch."""

    def __init__(
        self,
        module: Module,
        optimizer: Optimizer,
        criterion: Callable,
        privacy_engine: PrivacyEngine,
        train_loader: DataLoader,
        test_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float,
        batch_first: bool = True,
        loss_reduction_mean: bool = True,
        noise_generator: Generator = None,
        device: str = "cpu",
        **kwargs: Dict[str, Callable],
    ):
        """
        Parameters
        ----------
        module: torch.nn.Module
            A PyTorch neural network module instance.
        optimizer: torch.optim.Optimizer
            A PyTorch optimizer instance.
        criterion: Callable
            A function that takes predicted and actual values and returns a loss.
        privacy_engine: opacus.PrivacyEngine
            An Opacus PrivacyEngine instance.
        train_loader: torch.utils.data.DataLoader
            A PyTorch DataLoader instance for training data.
        test_loader: torch.utils.data.DataLoader
            A PyTorch DataLoader instance for test data.
        target_epsilon: float
            The privacy budget's epsilon
        target_delta: float
            The privacy budget's delta (probability of privacy leak)
        epochs: int
            The number of training epochs, to calculate noise multiplier to reach
            target epsilon and delta.
        max_grad_norm: float
            The maximum norm of the per-sample gradients. Any gradient with norm
            higher than this will be clipped to this value.
        batch_first: bool, default True
            Flag to indicate if the input tensor to the corresponding module
            has the first dimension representing the batch. If set to True,
            dimensions on input tensor are expected be ``[batch_size, ...]``,
            otherwise ``[K, batch_size, ...]``
        loss_reduction_mean: bool, default True
            Indicates if the loss reduction (for aggregating the gradients)
            is a mean (True) or sum (False) operation.
        noise_generator: torch.Generator(), default None
            PyTorch Generator instance used as a source of randomness for the noise.
        device: str, default "cpu"
            The device name to fit the model on.
        kwargs: Dict[str, Callable]
            A dictionary of metric functions to evaluate the model.
        """
        self.config = None
        self.criterion = criterion
        self.metric_functions = kwargs
        self.device = device
        self.privacy_engine = privacy_engine
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.epochs = epochs
        self.test_loader = test_loader
        loss_reduction = "mean" if loss_reduction_mean else "sum"
        (
            self.module,
            self.optimizer,
            self.train_loader,
        ) = self.privacy_engine.make_private_with_epsilon(
            module=module,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
        )

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set the PyTorch module parameters from a list of NumPy arrays."""
        params_dict = zip(self.module.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.module.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The local model parameters as a list of NumPy ndarrays.
        """
        return [param.cpu().numpy() for _, param in self.module.state_dict().items()]

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : List[numpy.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        self.set_parameters(parameters)
        self.config = config
        metrics = {}
        for _ in range(self.epochs):
            predictions = []
            actuals = []
            for x_train, y_train in tqdm(self.train_loader):
                num = y_train.size(0)
                if num > 0:
                    x_train = x_train.to(self.device)
                    y_train = y_train.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.module(x_train)
                    loss = self.criterion(outputs, y_train)
                    loss.backward()
                    self.optimizer.step()
                    predictions.extend(outputs.data)
                    actuals.extend(y_train)
            predictions = torch.stack(predictions, 0)
            actuals = torch.stack(actuals, 0)
            for name, fun in self.metric_functions.items():
                metrics.setdefault(name, [])
                metrics[name].append(fun(predictions, actuals))
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        accept = epsilon <= self.target_epsilon
        # metrics = {name: f(outputs, y_train) for name, f in self.metric_functions.items()}
        metrics["epsilon"] = epsilon
        metrics["accept"] = accept
        parameters = self.get_parameters() if accept else parameters
        return parameters, len(self.train_loader.dataset), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided weights using the locally held dataset.

        Parameters
        ----------
        parameters : List[np.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """
        self.set_parameters(parameters)
        self.config = config
        return test(
            self.module, self.criterion, self.test_loader, self.device, **self.metric_functions
        )


def test(
    module: Module,
    criterion: Callable,
    dataloader: DataLoader,
    device: str,
    **kwargs: Dict[str, Callable],
):
    """Validate the network on the test set."""
    predictions = []
    actuals = []
    num_examples = len(dataloader.dataset)
    loss = 0.0
    with torch.no_grad():
        module.to(device)
        for data in tqdm(dataloader):
            examples = data[0].to(device)
            labels = data[1].to(device)
            outputs = module(examples)
            loss += criterion(outputs, labels).item()
            predictions.extend(outputs.data)
            actuals.extend(labels)
    predictions = torch.stack(predictions, 0)
    actuals = torch.stack(actuals, 0)
    metrics = {name: f(predictions, actuals) for name, f in kwargs.items()}
    return loss / num_examples, num_examples, metrics
