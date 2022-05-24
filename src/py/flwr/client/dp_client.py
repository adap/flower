"""Differentially Private Client."""
from typing import Dict, List, Tuple

import numpy as np
from opacus import PrivacyEngine
from pytorch_lightning import LightningModule, Trainer

from flwr.common import Scalar

from .numpy_client import NumPyClient


class DPClient(NumPyClient):
    """Differentially private version of NumPyClient."""

    def __init__(self, module: LightningModule, trainer: Trainer, privacy_engine: PrivacyEngine):
        """
        Parameters
        ----------
        module: pytorch_lightning.LightningModule
            A PyTorch neural network module.
        trainer: pytorch_lightning.Trainer
            A PyTorch Lightning Trainer

        """
        self.module = module
        self.trainer = trainer
        # TODO: figure out how to attach PrivacyEngine to optimizer in a generic fashion
        self.privacy_engine = privacy_engine
        self.parameters = None
        self.config = None

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
        return

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
        return super().evaluate(parameters, config)
