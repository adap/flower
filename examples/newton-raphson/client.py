from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from utils import BaselineLoss, Newton, metric


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(
        self,
        net,
        trainloader,
        testloader,
        cpu_only,
        batch_size=4,
        l2_coeff=2.0,
    ) -> None:
        super().__init__()

        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.use_gpu = torch.cuda.is_available() and not (cpu_only)

        self._criterion = BaselineLoss()

        self.newton = Newton(batch_size, l2_coeff, list(self.net.parameters()))

    def get_parameters(self, config):
        gradients = self.newton.final_gradients

        assert gradients is not None

        gradients.append(self.newton.final_hessian)
        return gradients

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.net.train()

        l2_reg = self.newton.l2_reg(self.net.parameters())

        for X, y in self.trainloader:
            if self.use_gpu:
                X = X.cuda()
                y = y.cuda()

            y_pred = self.net(X)

            # Compute Loss
            loss = self._criterion(y_pred, y)

            # L2 regularization
            loss += l2_reg

            current_batch_size = len(X)

            self.newton.update_gradients_and_hessian(
                list(self.net.parameters()),
                loss,
                current_batch_size,
                len(self.trainloader.dataset),
            )

        assert self.newton.final_hessian is not None

        eigenvalues = np.linalg.eig(self.newton.final_hessian)[0].real
        if not (eigenvalues >= 0).all():
            raise ValueError(
                "Hessian matrix is not positive semi-definite, either the problem is not convex or due to numerical"
                " instability. It is advised to try to increase the l2_coeff. "
                f"Calculated eigenvalues are {eigenvalues.tolist()} and considered l2_coeff is {self.newton.l2_coeff}"
            )

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.net.eval()

        criterion = BaselineLoss()
        loss = 0.0

        with torch.no_grad():

            y_pred_final = []
            y_true_final = []

            for X, y in self.testloader:
                if self.use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                y_pred = self.net(X).detach().cpu()
                y = y.detach().cpu()
                loss += criterion(y_pred, y).item()
                y_pred_final.append(y_pred.numpy())
                y_true_final.append(y.numpy())

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)

        self.net.train()

        return (
            float(loss / len(self.testloader.dataset)),
            len(self.testloader.dataset),
            {"accuracy": float(metric(y_true_final, y_pred_final))},
        )
