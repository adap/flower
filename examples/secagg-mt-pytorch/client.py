import time

import numpy as np
from task import (
    DEVICE,
    IS_VALIDATION,
    Net,
    get_parameters,
    load_data,
    set_parameters,
    train,
)

import flwr as fl
from flwr.client.secure_aggregation import SecAggPlusHandler

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient, SecAggPlusHandler):
    # def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
    #     return get_parameters(net)

    def fit(self, parameters, config):
        if self._shared_state.test_drop:
            time.sleep(40)
        if IS_VALIDATION:
            return [np.zeros(10000)], 1, {}
        set_parameters(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        return get_parameters(net), len(trainloader.dataset), results

    # def evaluate(self, parameters, config):
    #     set_parameters(net, parameters)
    #     loss, accuracy = test(net, testloader)
    #     return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:9092",
    client=FlowerClient(),
    transport="grpc-rere",
)
