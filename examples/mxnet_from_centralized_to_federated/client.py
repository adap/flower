"""Flower client example using MXNet for MNIST classification."""

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import mxnet as mx
from mxnet import nd

import mxnet_mnist


# Flower Client
class MNISTClient(fl.client.NumPyClient):
    """Flower client implementing MNIST classification using MXNet."""

    def __init__(
        self,
        model: mxnet_mnist.model(),
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
        for val in self.model.collect_params(".*weight").values():
            p = val.data()
            # convert parameters from NDArray to Numpy Array required by Flower Numpy Client
            param.append(p.asnumpy())
        return param

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Collect model parameters and set new weight values
        params = zip(self.model.collect_params(".*weight").keys(), parameters)
        for key, value in params:
            self.model.collect_params().setattr(key, value)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        [accuracy, loss], num_examples = mxnet_mnist.train(
            self.model, self.train_data, epoch=2, device=self.device
        )
        results = {"accuracy": accuracy[1], "loss": loss[1]}
        return self.get_parameters(), num_examples, results

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[int, float, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        [accuracy, loss], num_examples = mxnet_mnist.test(
            self.model, self.val_data, device=self.device
        )
        print("Evaluation accuracy & loss", accuracy, loss)
        return (
            float(loss[1]),
            num_examples,
            {"accuracy": float(accuracy[1])},
        )


def main() -> None:
    """Load data, start MNISTClient."""

    # Set context to GPU or - if not available - to CPU
    DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]

    # Load data
    train_data, val_data = mxnet_mnist.load_data()

    # Load model (from centralized training)
    model = mxnet_mnist.model()

    # Do one forward propagation to initialize parameters
    init = nd.random.uniform(shape=(2, 784))
    model(init)

    # Start Flower client
    client = MNISTClient(model, train_data, val_data, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()
