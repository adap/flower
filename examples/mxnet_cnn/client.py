"""Flower client example using MXNet for MNIST classification."""

import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import mxnet as mx
from mxnet import nd
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
        model: mxnet_cnn.model(),
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
        print("Get Parameters")
        param = []
        for val in self.model.collect_params('.*weight').values():
            p = val.data()
            param.append(p.asnumpy())
        return param

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # arg_params: name of parameter
        # aux_params: value of parameters NDArray
        # self.model.initialize(parameters, force_reinit=True)
        print("Set Parameters")
        params = zip(self.model.collect_params('.*weight').keys(), parameters)
        for key, value in params:
            self.model.collect_params().setattr(key, value)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        print("Train Flower")
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        mxnet_cnn.train(self.model, self.train_data, epoch=3, device=self.device)
        return self.get_parameters(), self.train_data.batch_size

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = mxnet_cnn.test(self.model, self.val_data, device=self.device)
        print('Evaluation loss:', loss)
        print('Evaluation accuracy:', accuracy)
        print('NUmber of examples',self.val_data.batch_size)
        return self.val_data.batch_size, float(loss), float(accuracy)


def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    print("Define Device")
    DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    #DEVICE = [mx.cpu()]
    print("Load data")
    train_data, val_data = mxnet_cnn.load_data()
    print("define model")
    model = mxnet_cnn.model()
    print("Make 1 forward")
    init = nd.random.uniform(shape=(2, 784))
    model(init)

    # Start client
    client = MNISTClient(model, train_data, val_data, DEVICE)
    print("Fit done")
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
