"""Differentially Private Client."""
from typing import Dict, List, Tuple

import numpy as np
from opacus import PrivacyEngine
from torch import Generator
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from flwr.client.numpy_client import NumPyClient
from flwr.common import Scalar


class DPClient(NumPyClient):
    """Differentially private version of NumPyClient."""

    def __init__(
        self,
        module: Module,
        optimizer: Optimizer,
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
    ):
        """
        Parameters
        ----------
        module: torch.nn.Module
            A PyTorch neural network module instance.
        optimizer: torch.optim.Optimizer
            A PyTorch optimizer instance.
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
        """
        self.parameters = None
        self.config = None
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

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The local model parameters as a list of NumPy ndarrays.
        """
        return self.parameters

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
        self.parameters = parameters
        self.config = config
        # TODO: write a train loop
        # TODO: update self.parameters after fitting but only if target_epsilon not exceeded (accept is True)
        # TODO: add any other metrics we need to report back to the server
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        accept = epsilon <= self.target_epsilon
        metrics = {"epsilon": epsilon, "accept": accept}
        return self.parameters, len(self.train_loader), metrics

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
        # TODO: test loop
        return super().evaluate(parameters, config)
