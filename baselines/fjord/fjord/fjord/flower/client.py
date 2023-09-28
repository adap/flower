import random
from collections import OrderedDict
from typing import List, Dict, Optional, Union, Any, Tuple

import numpy as np
import flwr as fl
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer
import torchvision
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from fjord.data.fl_cifar10 import FLCifar10, FLCifar10Client
from fjord.od.samplers import ODSampler
from fjord.od.layers import ODConv2d, ODLinear, ODBatchNorm2d
from fjord.utils.utils import get_net, save_model
from fjord.utils.logger import Logger


CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
FJORD_CONFIG_TYPE = Dict[Union[str, float], List[Union[Dict[str, int], str]]]


def get_lr_scheduler(optimiser: Optimizer,
                     total_epochs: int,
                     method: Optional[str] = 'static',
                     ) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get the learning rate scheduler.
    :param optimiser: The optimiser for which to get the scheduler.
    :param total_epochs: The total number of epochs.
    :param method: The method to use for the scheduler.
        Supports static and cifar10.
    :return: The learning rate scheduler.
    """
    if method == 'static':
        return MultiStepLR(optimiser, [total_epochs + 1])
    elif method == 'cifar10':
        return MultiStepLR(optimiser,
                           [int(0.5 * total_epochs), int(0.75 * total_epochs)],
                           gamma=0.1)
    raise ValueError(f"{method} scheduler not currently supported.")


def train(net: Module, trainloader: DataLoader, know_distill: bool,
          max_p: float, current_round: int, total_rounds: int,
          p_s: List[float], epochs: int,
          train_config: Dict[str, fl.common.Scalar]) -> float:
    """Train the model on the training set.
    :param net: The model to train.
    :param trainloader: The training set.
    :param know_distill: Whether the model being trained
        uses knowledge distillation.
    :param max_p: The maximum p value.
    :param current_round: The current round of training.
    :param total_rounds: The total number of rounds of training.
    :param p_s: The p values to use for training.
    :param epochs: The number of epochs to train for.
    :param train_config: The training configuration.
    :return: The loss on the training set.
    """
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    if train_config.optimiser == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=train_config.lr,
                                    momentum=train_config.momentum,
                                    nesterov=train_config.nesterov,
                                    weight_decay=train_config.weight_decay,)
    else:
        raise ValueError(f"Optimiser {train_config.optimiser} not supported")
    lr_scheduler = get_lr_scheduler(optimizer, total_rounds,
                                    method=train_config.lr_scheduler)
    for _ in range(current_round):
        lr_scheduler.step()

    sampler = ODSampler(
        p_s=p_s,
        max_p=max_p,
        model=net,)
    max_sampler = ODSampler(
        p_s=[max_p],
        max_p=max_p,
        model=net,)

    loss = 0.0
    samples = 0
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            target = labels.to(device)
            images = images.to(device)
            batch_size = images.shape[0]
            if know_distill:
                full_output = net(
                    images.to(device), sampler=max_sampler)
                full_loss = criterion(full_output, target)
                full_loss.backward()
                target = full_output.detach().softmax(dim=1)
            partial_loss = criterion(
                net(images, sampler=sampler), target)
            partial_loss.backward()
            optimizer.step()
            loss += partial_loss.item() * batch_size
            samples += batch_size

    return loss / samples


def test(net: Module, testloader: DataLoader, p_s: List[float]
         ) -> Tuple[List[float], List[float]]:
    """Validate the model on the test set.
    :param net: The model to validate.
    :param testloader: The test set.
    :param p_s: The p values to use for validation.
    :return: The loss and accuracy on the test set."""
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    net.eval()

    for p in p_s:
        correct, loss = 0, 0.0
        p_sampler = ODSampler(
            p_s=[p],
            max_p=p,
            model=net,)

        with torch.no_grad():
            for images, labels in tqdm(testloader):
                outputs = net(images.to(device), sampler=p_sampler)
                labels = labels.to(device)
                loss += criterion(outputs, labels).item() * images.shape[0]
                correct += (torch.max(
                    outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        losses.append(loss / len(testloader.dataset))
        accuracies.append(accuracy)

    return losses, accuracies


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get the transforms for the CIFAR10 dataset.
    :return: The transforms for the CIFAR10 dataset.
    """
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

    return transform_train, transform_test


def load_data(path: str, cid: int, train_bs: int,
              seed: int, eval_bs: int = 1024
              ) -> Tuple[DataLoader, DataLoader]:
    """
    Load the CIFAR10 dataset.
    :param path: The path to the dataset.
    :param cid: The client ID.
    :param train_bs: The batch size for training.
    :param seed: The seed to use for the random number generator.
    :param eval_bs: The batch size for evaluation.
    :return: The training and test sets.
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    transform_train, transform_test = get_transforms()

    fl_dataset = FLCifar10(
        root=path, train=True, download=True,
        transform=transform_train)

    trainset = FLCifar10Client(fl_dataset, client_id=cid)
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True,
        transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=train_bs, shuffle=True,)
    #  worker_init_fn=seed_worker, generator=g)  # BUG
    test_loader = DataLoader(testset, batch_size=eval_bs)

    return train_loader, test_loader


def get_layer_from_state_dict(model: Module, state_dict_key: str
                              ) -> Module:
    """
    Get the layer corresponding to the given state dict key.
    :param model: The model.
    :param state_dict_key: The state dict key.
    :return: The module corresponding to the given state dict key.
    """
    keys = state_dict_key.split('.')
    module = model
    # The last keycorresponds to the parameter name
    # (e.g., weight or bias)
    for key in keys[:-1]:
        module = getattr(module, key)
    return module


def net_to_state_dict_layers(net: Module) -> List[Module]:
    """
    Get the state_dict of the model.
    :param net: The model.
    :return: The state_dict of the model."""
    layers = []
    for key, val in net.state_dict().items():
        layer = get_layer_from_state_dict(net, key)
        layers.append(layer)
    return layers


def get_agg_config(net: Module, trainloader: DataLoader, p_s: List[float]
                   ) -> FJORD_CONFIG_TYPE:
    """
    Get the aggregation configuration of the model.
    :param net: The model.
    :param trainloader: The training set.
    :param p_s: The p values used
    :return: The aggregation configuration of the model.
    """
    Logger.get().info("Constructing OD model configuration for aggregation.")
    device = next(net.parameters()).device
    images, _ = next(iter(trainloader))
    images = images.to(device)
    layers = net_to_state_dict_layers(net)
    # init min dims in networks
    config = {p: [{} for _ in layers] for p in p_s}
    config['layer'] = []
    config['layer_p'] = []
    with torch.no_grad():
        for p in p_s:
            max_sampler = ODSampler(
                p_s=[p],
                max_p=p,
                model=net,)
            net(images, sampler=max_sampler)
            for i, layer in enumerate(layers):
                if isinstance(layer, ODConv2d) or isinstance(
                            layer, ODLinear):
                    config[p][i]['in_dim'] = layer.last_input_dim
                    config[p][i]['out_dim'] = layer.last_output_dim
                elif isinstance(layer, ODBatchNorm2d):
                    config[p][i]['in_dim'] = None
                    config[p][i]['out_dim'] = layer.p_to_num_features[p]
                elif isinstance(layer, torch.nn.BatchNorm2d):
                    pass
                else:
                    raise ValueError(
                        f"Unsupported layer {layer.__class__.__name__}"
                    )
    for layer in layers:
        config['layer'].append(layer.__class__.__name__)
        if hasattr(layer, 'p'):
            config['layer_p'].append(layer.p)
        else:
            config['layer_p'].append(None)
    return config


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    """
    Flower client training on CIFAR-10.
    """
    def __init__(self, cid: int, model_name: str, model_path: str,
                 data_path: str, know_distill: bool, max_p: float,
                 p_s: List[float], train_config: Dict[str, fl.common.Scalar],
                 fjord_config: Dict[str, fl.common.Scalar],
                 log_config: Dict[str, str],
                 device: torch.device, seed: int) -> None:
        """
        Initialise the client.
        :param cid: The client ID.
        :param model_name: The model name.
        :param model_path: The path to save the model.
        :param data_path: The path to the dataset.
        :param know_distill: Whether the model uses knowledge distillation.
        :param max_p: The maximum p value.
        :param p_s: The p values to use for training.
        :param train_config: The training configuration.
        :param fjord_config: The configuration for Fjord.
        :param log_config: The logging configuration.
        :param device: The device to use.
        :param seed: The seed to use for the random number generator.
        """
        Logger.setup_logging(**log_config)
        self.cid = cid
        self.p_s = p_s
        self.net = get_net(model_name, p_s, device)
        self.trainloader, self.valloader = load_data(
            data_path, int(cid), train_config.batch_size, seed)

        self.know_distill = know_distill
        self.max_p = max_p
        self.fjord_config = fjord_config
        self.train_config = train_config
        self.model_path = model_path

    def get_parameters(self, config: Dict[str, fl.common.Scalar]
                       ) -> List[np.ndarray]:
        """
        Get the parameters of the model to return to the server.
        :param config: The configuration.
        :return: The parameters of the model.
        """
        Logger.get().info(f"Getting parameters from client {self.cid}")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def net_to_state_dict_layers(self) -> List[Module]:
        """
        Model to state dict layers.
        """
        return net_to_state_dict_layers(self.net)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sets the parameters of the model.
        :param parameters: The parameters of the model.
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[Tensor],
            config: Dict[str, fl.common.Scalar]
            ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train the model on the training set.
        :param parameters: The parameters of the model.
        :param config: The train configuration.
        :return: The parameters of the model, the number of
            samples used for training, and the training metrics
        """
        Logger.get().info(
            f"Training on client {self.cid} for round "
            f"{config['current_round']}/{config['total_rounds']}")

        original_parameters = deepcopy(parameters)

        self.set_parameters(parameters)
        self.train_config.lr = config['lr']

        loss = train(
            self.net, self.trainloader,
            self.know_distill, self.max_p,
            p_s=self.p_s, epochs=self.train_config.local_epochs,
            current_round=config['current_round'],
            total_rounds=config['total_rounds'],
            train_config=self.train_config)

        final_parameters = self.get_parameters(config={})

        return final_parameters, \
            len(self.trainloader.dataset), {
                'max_p': self.max_p,
                'p_s': self.p_s,
                'fjord_config': self.fjord_config,
                'original_parameters': original_parameters,
                'loss': loss,
                }

    def evaluate(self, parameters: List[np.ndarray],
                 config: Dict[str, fl.common.Scalar]
                 ) -> Tuple[float, int, Dict[str, float]]:
        """
        Validate the model on the test set.
        :param parameters: The parameters of the model.
        :param config: The eval configuration.
        :return: The loss on the test set, the number of samples
            used for evaluation, and the evaluation metrics.
        """
        Logger.get().info(
            f"Evaluating on client {self.cid} for round "
            f"{config['current_round']}/{config['total_rounds']}")

        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.max_p)
        save_model(self.net, self.model_path, cid=self.cid)

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
