import time

import numpy as np
from task import (
    DEVICE,
    IS_VALIDATION,
    NUM_ITERATIONS,
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


# Define Flower client with the SecAgg/SecAgg+ protocol
class FlowerClient(fl.client.NumPyClient, SecAggPlusHandler):
    def fit(self, parameters, config):
        # Force a significant delay for teshing purposes
        if self._shared_state.sid == 0:
            time.sleep(40)
        if IS_VALIDATION:
            return [np.zeros(10000)], 1, {}
        set_parameters(net, parameters)
        results = train(
            net, trainloader, testloader, num_iterations=NUM_ITERATIONS, device=DEVICE
        )
        return (
            get_parameters(net),
            len(trainloader.batch_size * NUM_ITERATIONS),
            results,
        )


# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:9092",
    client=FlowerClient(),
    transport="grpc-rere",
)
