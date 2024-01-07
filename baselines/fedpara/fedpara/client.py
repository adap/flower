"""Client for FedPara."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Optional
import copy,os
import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from fedpara.models import train,test
import logging

class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        train_loader: DataLoader,
        device: str,
        num_epochs: int,
    ):  # pylint: disable=too-many-arguments
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.num_epochs = num_epochs

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the network on the training set."""
        self.set_parameters(parameters)

        train(
            self.net,
            self.train_loader,
            self.device,
            epochs=self.num_epochs,
            hyperparams=config,
            epoch=int(config["curr_round"]),
        )

        return (
            self.get_parameters({}),
            len(self.train_loader),
            {},
        )

class PFlowerClient(fl.client.NumPyClient):
    """personalized Flower Client"""
    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        num_epochs: int,
        state_path: str,
        algorithm: str,
    ): 
        
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.state_path = state_path
        self.algorithm = algorithm

    def get_keys_state_dict(self, mode:str="local")->list[str]:
        match self.algorithm:
            case "fedper":
                if mode == "local":
                    return list(filter(lambda x: 'fc2' in x,self.net.state_dict().keys()))
                elif mode == "global":
                    return list(filter(lambda x: 'fc1' in x,self.net.state_dict().keys()))
            case "pfedpara":
                if mode == "local":
                    return list(filter(lambda x: 'w2' in x,self.net.state_dict().keys()))
                elif mode == "global":
                    return list(filter(lambda x: 'w1' in x,self.net.state_dict().keys()))
            case _:
                raise NotImplementedError(f"algorithm {self.algorithm} not implemented")
            
            
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        model_dict = self.net.state_dict()
        #TODO: overwrite the server private parameters
        for k in self.private_server_param.keys():
            model_dict[k] = self.private_server_param[k]
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        self.private_server_param: Dict[str, torch.Tensor] = {}
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.private_server_param = {k:state_dict[k] for k in self.get_keys_state_dict(mode="local")}
        self.net.load_state_dict(state_dict, strict=True)
        if os.path.isfile(self.state_path):
            # only overwrite global parameters
            with open(self.state_path, 'rb') as f:
                model_dict = self.net.state_dict()
                state_dict = torch.load(f)
                for k in self.get_keys_state_dict(mode="global"):
                    model_dict[k] = state_dict[k]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the network on the training set."""
        self.set_parameters(parameters)
        print(f"Client {self.cid} Training...")

        train(
            self.net,
            self.train_loader,
            self.device,
            epochs=self.num_epochs,
            hyperparams=config,
            epoch=config["curr_round"],
        )
        if self.state_path is not None:
            with open(self.state_path, 'wb') as f:
                torch.save(self.net.state_dict(), f)

        return (
            self.get_parameters({}),
            len(self.train_loader),
            {}, 
        )
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """Evaluate the network on the test set."""
        self.set_parameters(parameters)
        print(f"Client {self.cid} Evaluating...")
        self.net.to(self.device)
        loss, accuracy = test(self.net, self.test_loader, device=self.device)
        return loss, len(self.test_loader), {"accuracy": accuracy}
     

def gen_client_fn(
    train_loaders: List[DataLoader],
    model: DictConfig,
    num_epochs: int,
    args: Dict,
    test_loader: Optional[List[DataLoader]]=None,
    state_path: Optional[str]=None,    
) -> Callable[[str], fl.client.NumPyClient]:
    """Return a function which creates a new FlowerClient for a given cid."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        """Create a new FlowerClient for a given cid."""
        cid = int(cid)
        if args['algorithm'].lower() == "pfedpara" or args['algorithm'] == "fedper":
            cl_path = f"{state_path}/client_{cid}.pth"
            return PFlowerClient(
                cid=cid,
                net=instantiate(model).to(args["device"]),
                train_loader=train_loaders[cid],
                test_loader=copy.deepcopy(test_loader),
                device=args["device"],
                num_epochs=num_epochs,
                state_path=cl_path,
                algorithm=args['algorithm'].lower(),
            )
        else:
            return FlowerClient(
                cid=cid,
                net=instantiate(model).to(args["device"]),
                train_loader=train_loaders[cid],
                device=args["device"],
                num_epochs=num_epochs,
            )
       
    return client_fn
