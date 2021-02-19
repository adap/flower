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

import mxnet_cnn


# Flower Client
class MNISTClient(fl.client.NumPyClient):
    """Flower client implementing MNIST classification using
    MXNet."""

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
        # Return model parameters as a list of NumPy Arrays
        param = []
        for val in self.model.collect_params('.*weight').values():
            p = val.data()
            # convert parameters from NDArray to Numpy Array required by Flower Numpy Client
            param.append(p.asnumpy())
        return param

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Collect model parameters and set new weight values
        params = zip(self.model.collect_params('.*weight').keys(), parameters)
        for key, value in params:
            self.model.collect_params().setattr(key, value)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        mxnet_cnn.train(self.model, self.train_data, epoch=1, device=self.device)
        return self.get_parameters(), self.train_data.batch_size

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = mxnet_cnn.test(self.model, self.val_data, device=self.device)
        return self.val_data.batch_size, float(loss), float(accuracy)


def main() -> None:
    """Load data, start CifarClient."""

    # Setup context to GPU and if not available to CPU
    DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    # Load data
    train_data, val_data = mxnet_cnn.load_data()
    # Define model from centralized training
    model = mxnet_cnn.model()
    # Make one forward propagation to initialize parameters
    init = nd.random.uniform(shape=(2, 784))
    model(init)

    # Start Flower client
    client = MNISTClient(model, train_data, val_data, DEVICE)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
