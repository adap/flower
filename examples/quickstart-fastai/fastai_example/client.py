import warnings
from collections import OrderedDict

import torch
from fastai.vision.all import *

import flwr as fl


warnings.filterwarnings("ignore", category=UserWarning)

# Download MNIST dataset
path = untar_data(URLs.MNIST)

# Load dataset
dls = ImageDataLoaders.from_folder(
    path, valid_pct=0.5, train="training", valid="testing", num_workers=0
)

# Define model
learn = vision_learner(dls, squeezenet1_1, metrics=error_rate)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in learn.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(learn.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        learn.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        learn.fit(1)
        return self.get_parameters(config={}), len(dls.train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, error_rate = learn.validate()
        return loss, len(dls.valid), {"accuracy": 1 - error_rate}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)
