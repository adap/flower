"""Flower client example using MXNet for MNIST classification."""

import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import mxnet as mx
from mxnet.gluon import ParameterDict


import mxnet_cnn

# pylint: disable=no-member
DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
# pylint: enable=no-member

# Flower Client
class MNISTClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: mxnet_cnn.Net,
        train_data: mx.io.NDArrayIter,
        val_data: mx.io.NDArrayIter,
        device: mx.context,
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.device = device

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        # return [val.data() for val in self.model.collect_params().values()]
        # model_params = [val.data() for val in self.model.collect_params().values()]
        return self.model.collect_params()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # arg_params: name of parameter
        # aux_params: value of parameters NDArray
        # self.model.initialize(parameters, force_reinit=True)
        params = zip(self.model.collect_params().keys(), parameters)
        for key, value in params:
            self.model.collect_params().setattr(key, value)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        mxnet_cnn.train(self.model, self.train_data, epoch=2, device=self.device)
        return self.get_parameters(), len(list(self.train_data))

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = mxnet_cnn.test(self.model, self.val_data, device=self.device)
        return len(list(self.val_data)), float(loss), float(accuracy)


def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    model = mxnet_cnn.Net()
    train_data, val_data = mxnet_cnn.load_data()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=DEVICE)

    # Start client
    client = MNISTClient(model, train_data, val_data, DEVICE)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
