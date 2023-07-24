
import argparse
from time import time
from collections import OrderedDict

import flwr as fl
import torch

from model_and_dataset import Net, load_data, train, test

parser = argparse.ArgumentParser(description="In-FileSystem Stateful Flower Clients")

parser.add_argument("--client_id", type=str, required=True, help="An arbitrary identifier for your client so multiple clients in the same system do not interfere with each others' state")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# #############################################################################
# Run Federated Learning with Flower using an In-FileSystem stateful client
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client using In-FileSystem state.
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_identifier) -> None:
        super().__init__()

        self.state = fl.client.InFileSystemClientState()
        # we want each client to have its own state in the file system
        # therefore we should provide the state object with a unique path
        # one way to achieve this is by using a unique identifier for each
        # client. Note this is only needed if multiple clients co-exist in 
        # the same system (as it might happen when you run this example)
        state_path = f'./client_states/{client_identifier}'
        self.state.setup(state_path, create_directory=True)
        state = self.state.fetch(from_disk=True)
        print(f"Initial state: {state}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        state = self.state.fetch(from_disk=True)
        print(f"Current state: {state}")
        t_start = time()
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        t_end = time() - t_start
        self.state.update({'fit_time': t_end}, to_disk=True)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(args.client_id),
    )
