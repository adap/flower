from typing import Dict
import flwr as fl
from flwr.common import NDArrays, Scalar

from task import (
    Net,
    DEVICE,
    load_data,
    get_parameters,
    set_parameters,
    train,
    test,
)


restful_mode = False
# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(net)

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        return get_parameters(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
if restful_mode:
    fl.client.start_numpy_client(
        server_address="http://0.0.0.0:9093",
        client=FlowerClient(),
        rest=True,
        transport="rest",
    )
else:
    fl.client.start_numpy_client(
        server_address="0.0.0.0:9092",
        client=FlowerClient(),
        transport="grpc-rere",
    )
