import warnings
from collections import OrderedDict

import torch
from fastai.vision.all import *

import flwr as fl


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
defaults.device = torch.device("cpu")

path = untar_data(URLs.PETS)
files = get_image_files(path / "images")


def label_func(f):
    return f[0].isupper()


dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.numpy() for _, val in learn.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(learn.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        learn.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        learn.fine_tune(1)
        return self.get_parameters(config={}), len(dls.train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, error_rate = learn.validate()
        return loss, len(dls.valid), {"accuracy": 1 - error_rate}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
