
#import matplotlib.pyplot as plt
#from sklearn.metrics import accuracy_score
import torch 

#from sklearn.tree import DecisionTreeClassifier

from collections import OrderedDict

import flwr as fl

from .model import MulticlassClassification, train, validation

#from torchvision.transforms import Compose, Normalize, ToTensor
#import sys
#path = "/home/najet/Desktop/ICS/hyb/res2/client3.txt"
#sys.stdout = open(path, 'w')

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, valloader,num_features, total_classes, num_classes_cl, epoch, lr, server_round) -> None:
        super().__init__()

        self.id=cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.epoch=epoch
        self.lr=lr
        # a model that is randomly initialised at first
        self.model = MulticlassClassification(num_features,total_classes)
        self.num_classes_cl=num_classes_cl
        self.total_classes=total_classes
        self.server_round=server_round
        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def get_parameters(self, config):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
       
        self.set_parameters(parameters)
        train(self.model, self.trainloader,self.valloader, self.epoch,self.lr)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"coef":self.num_classes_cl/self.total_classes,"cid":self.id}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        print("Classification report client id", self.id)
        loss, accuracy = validation(self.model, self.valloader, config["current_round"], self.server_round)
        print("client id", self.id,{"round": config["current_round"]}, {"loss": loss}, {"accuracy": accuracy})
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "cid":self.id}

def generate_client_fn(trainloaders, valloaders, num_features,total_classes, num_classes_cl, epoch, lr,server_round):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
       
        return FlowerClient(
            int(cid),
            trainloaders[int(cid)],
            valloaders[int(cid)],
            num_features,
            total_classes,
            num_classes_cl[int(cid)],
            epoch,
            lr,
            server_round,
        )

    # return the function to spawn client
    return client_fn


