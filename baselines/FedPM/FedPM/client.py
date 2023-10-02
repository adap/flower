"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import flwr

from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np
import torch
from scipy.stats import bernoulli
from logging import INFO
from models import load_model

import torch.nn as nn
from flwr.common.logger import log
from flwr.common import (
    FitIns,
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    GetParametersIns,
    GetParametersRes,
    EvaluateIns,
    EvaluateRes
)

# TO CHECK
from models import get_parameters, set_parameters


class FedPMClient(flwr.client.Client):
    def __init__(
            self,
            params: Dict,
            client_id: int,
            train_data_loader: DataLoader,
            test_data_loader: DataLoader,
            device='cpu',
    ) -> None:

        self.params = params
        self.client_id = client_id
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.local_model = load_model(self.params).to(self.device)
        self.epsilon = 0.01

    def sample_mask(self, mask_probs: Dict) -> List[np.ndarray]:
        sampled_mask = []
        with torch.no_grad():
            for layer_name, layer in mask_probs.items():
                if 'mask' in layer_name:
                    theta = torch.sigmoid(layer).cpu().numpy()
                    updates_s = bernoulli.rvs(theta)
                    updates_s = np.where(updates_s == 0, self.epsilon, updates_s)
                    updates_s = np.where(updates_s == 1, 1 - self.epsilon, updates_s)
                    sampled_mask.append(updates_s)
                else:
                    sampled_mask.append(layer.cpu().numpy())

        return sampled_mask

    def fit(
            self,
            fitins: FitIns
    ) -> FitRes:
        self.set_parameters(fitins.parameters)
        mask_probs = self.train_fedpm(
            model=self.local_model,
            trainloader=self.train_data_loader,
            iter_num=fitins.config.get('iter_num'),
            params=self.params,
        )
        sampled_mask = self.sample_mask(mask_probs)
        parameters = ndarrays_to_parameters(sampled_mask)
        status = Status(code=Code.OK, message="Success")

        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=len(self.train_data_loader),
            metrics={}
        )

    def evaluate(
            self,
            ins: EvaluateIns
    ) -> EvaluateRes:

        loss_fn = torch.nn.CrossEntropyLoss()
        self.set_parameters(ins.parameters)
        loss, accuracy = 0, 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_id, (inputs, labels) in enumerate(self.test_data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss += loss_fn(outputs, labels)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
            accuracy = 100 * correct / total
        status = Status(Code.OK, message="Success")
        log(
            INFO,
            "Client %s Accuracy: %f   Loss: %f",
            self.client_id,
            accuracy,
            loss
        )
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.test_data_loader),
            metrics={"accuracy": float(accuracy),
                     "loss": float(loss)}
        )

    def train_fedpm(
            self,
            model: nn.Module,
            trainloader: DataLoader,
            iter_num: int,
            params: Dict,
            loss_fn=nn.CrossEntropyLoss(reduction='mean'),
            optimizer: torch.optim.Optimizer = None
    ) -> Dict:
        """
            Compute local epochs, the training strategies depends on the adopted model.
        """

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('strategy').get('fedpm').get('local_lr'))

        for epoch in range(params.get('strategy').get('fedpm').get('local_epochs')):
            running_loss = 0
            total = 0
            correct = 0
            for batch_idx, (train_x, train_y) in enumerate(trainloader):
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                total += train_x.size(0)
                optimizer.zero_grad()
                y_pred = model(train_x)
                loss = loss_fn(y_pred, train_y)
                running_loss += loss.item()
                _, pred_y = torch.max(y_pred.data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def get_parameters(
            self,
            ins: GetParametersIns = None
    ) -> GetParametersRes:
        ndarrays = get_parameters(self.local_model)
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )

    def set_parameters(self, parameters):
        set_parameters(
            self.local_model,
            parameters_to_ndarrays(parameters)
        )
