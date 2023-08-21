import random
import argparse
from time import time
from collections import OrderedDict

import wandb
import flwr as fl
import torch
from torch.utils.data import DataLoader

from utils import Net, test, prepare_dataset

parser = argparse.ArgumentParser(description="Flower + W&B + PyTorch")

parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client it. Will be used to assign the i-th partition to the client (default = 0)",
)
parser.add_argument(
    "--rounds", type=int, default=20, help="Number of FL rounds (default = 20)"
)


TOTAL_CLIENTS = 50  # the dataset will be partitioned in this many clients
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, optimizer, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(DEVICE)), labels.to(DEVICE))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    return total_loss / len(trainloader.dataset)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainset, valset):
        super().__init__()
        self.cid = cid
        self.model = Net().to(DEVICE)
        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        self.valloader = DataLoader(valset, batch_size=32, shuffle=False)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Fetch details sent by the server to connect to the same W&B run
        project, group, round = config["project"], config["group"], config["round"]
        wandb.init(project=project, name=f"client-{self.cid}", group=group)
        t_start = time()

        # do fit() as usual
        self.set_parameters(parameters)
        # to make the W&B plots more interesting, let's add some "noise" to the quality
        # of the training.
        lr = random.random() * 0.05
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # let's add some randomness in the number of local epochs too
        # this will simulate some clients take longer than others to train
        epochs = random.randint(1, 5)
        train_loss = train(self.model, optim, self.trainloader, epochs=epochs)

        # Log something to W&B
        wandb.log({"train_loss": train_loss, "local_train_time": time() - t_start})
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, DEVICE)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def main():
    # Parse input arguments
    args = parser.parse_args()

    assert (
        args.cid < TOTAL_CLIENTS
    ), f"Ensure the specified cid ({args.cid}) is < {TOTAL_CLIENTS}"

    # fabricate dataset for client (using user-suplied cid)
    clients_trainset, clients_valset, _ = prepare_dataset(TOTAL_CLIENTS)

    def client_fn():
        return FlowerClient(
            args.cid, clients_trainset[args.cid], clients_valset[args.cid]
        )

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client_fn(),
    )


if __name__ == "__main__":
    main()
