import numpy as np
import torch
import torch.nn as nn
from flamby.datasets.fed_heart_disease import FedHeartDisease
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader as dl


class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.bce(input, target)


def get_data(cid, train=True):
    return dl(
        FedHeartDisease(center=cid, train=train),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )


def get_dataset(cid, train=True):
    return FedHeartDisease(center=cid, train=train)


def validate(net, testloader, use_gpu):
    """Validate the model on the test set."""
    if use_gpu:
        net = net.cuda()
    net.eval()
    criterion = BaselineLoss()
    loss = 0.0
    with torch.no_grad():
        y_pred_final = []
        y_true_final = []
        for X, y in testloader:
            if use_gpu:
                X = X.cuda()
                y = y.cuda()
            y_pred = net(X).detach().cpu()
            y = y.detach().cpu()
            loss += criterion(y_pred, y).item()
            y_pred_final.append(y_pred.numpy())
            y_true_final.append(y.numpy())

        y_true_final = np.concatenate(y_true_final)
        y_pred_final = np.concatenate(y_pred_final)
    net.train()
    return loss / len(testloader.dataset), metric(y_true_final, y_pred_final)


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).mean()
    except ValueError:
        return np.nan


class Newton:
    def __init__(self, batch_size, l2_coeff, params) -> None:
        self.batch_size = batch_size
        self.l2_coeff = l2_coeff

        self.final_gradients, self.final_hessian, self.n_samples_done = (
            self.initialize_gradients_and_hessian(params)
        )

    def jacobian(self, tensor_y, params, create_graph=False):
        jacobian = []
        flat_y = torch.cat([t.reshape(-1) for t in tensor_y])
        for y in flat_y:
            for param in params:
                if param.requires_grad:
                    (gradient,) = torch.autograd.grad(
                        y, param, retain_graph=True, create_graph=create_graph
                    )
                    jacobian.append(gradient)

        return jacobian

    def hessian_shape(self, second_order_derivative):
        hessian = torch.cat([t.reshape(-1) for t in second_order_derivative])

        assert self.final_hessian is not None

        return hessian.reshape(self.final_hessian.shape)

    def compute_gradients_and_hessian(self, params, loss):
        gradients = self.jacobian(loss[None], params, create_graph=True)
        second_order_derivative = self.jacobian(gradients, params)

        hessian = self.hessian_shape(second_order_derivative)

        return gradients, hessian

    def l2_reg(self, params):
        # L2 regularization
        l2_reg = 0
        for param in params:
            l2_reg += self.l2_coeff * torch.sum(param**2) / 2
        return l2_reg

    def initialize_gradients_and_hessian(self, params):
        number_of_trainable_params = sum(p.numel() for p in params if p.requires_grad)

        n_samples_done = 0

        final_gradients = [torch.zeros_like(p).numpy() for p in params]
        final_hessian = np.zeros(
            [number_of_trainable_params, number_of_trainable_params]
        )

        return final_gradients, final_hessian, n_samples_done

    def update_gradients_and_hessian(
        self, params, loss, current_batch_size, trainloader_size
    ):
        gradients, hessian = self.compute_gradients_and_hessian(params, loss)

        self.n_samples_done += current_batch_size

        batch_coefficient = current_batch_size / trainloader_size

        assert self.final_gradients is not None

        self.final_hessian += hessian.cpu().detach().numpy() * batch_coefficient
        self.final_gradients = [
            sum(final_grad, grad.cpu().detach().numpy() * batch_coefficient)
            for final_grad, grad in zip(self.final_gradients, gradients)
        ]
