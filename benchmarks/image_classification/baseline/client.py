import torch
from torch.utils.data import DataLoader
import flwr as fl

from utils import train, set_params
from model import Net


# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset):
        self.trainset = trainset

        # Instantiate model
        self.model = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]
        lr, momentum = config["lr"], config["momentum"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # Train
        loss, accuracy = train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        metric = {"train_loss": loss, "train_acc": accuracy}

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), metric
