from collections import OrderedDict

import flwr as fl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def main():
    class MnistModel(nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            # hidden layer
            self.linear1 = nn.Linear(in_size, hidden_size)

            # out layer
            self.linear2 = nn.Linear(hidden_size, out_size)

        def forward(self, xb):
            # flatten the tensors to size 100x784
            xb = xb.view(xb.size(0), -1)
            # get intermediate outputs from the hidden layer

            out = self.linear1(xb)

            # apply activation function ReLU

            out = F.relu(out)

            # get the prediction using the output layer

            out = self.linear2(out)

            return out


    input_size = 784
    num_classes = 10

    model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

    train_dl, valid_dl = load_data()

    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    #Flower Client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]


        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)

            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)


        def fit(self, parameters, config):
            self.set_parameters(parameters)

            current_parameters = self.get_parameters()

            train(1, 0.5, model, F.cross_entropy, train_dl, valid_dl, accuracy)

            new_parameters = self.get_parameters()

            current_parameters = np.array(current_parameters, dtype=object)
            new_parameters = np.array(new_parameters, dtype=object)

            difference_in_parameters = np.subtract(new_parameters, current_parameters)

            result=[]

            for x in difference_in_parameters:
                hold = np.array(x)
                result.append(hold)

            return result, len(train_dl), {}


        def evaluate(self, parameters, config):
            self.set_parameters(parameters)

            results = test(model, F.cross_entropy, valid_dl, accuracy)
            loss, total, val_metric = results

            return float(loss), len(valid_dl), {"accuracy": float(val_metric)}


    fl.client.start_numpy_client("[::]:8080", client=MnistClient())




def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)




def train(epochs, lr, model, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None):
    losses, metrics = [], []

    if opt_fn is None: opt_fn = torch.optim.SGD

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss_batch(model, loss_fn, xb, yb, opt)



def test(model, loss_func, valid_dl, metric=None):
    with torch.no_grad():
        # for each batch calculate the loss, batch size, accuracy
        results = [loss_batch(model, loss_func, xb, yb, metric=metric) for xb, yb in valid_dl]

        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

        return avg_loss, total, avg_metric


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None

    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def to_device(data, device):
    if isinstance (data, (list, tuple)):
        return [to_device(x , device ) for x in data]
    return data.to(device, non_blocking=True)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')



def split_indices(n, val_pct):
    n_val = int(n * val_pct)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[: n_val]



def load_data():
    dataset = MNIST('./dataset', download=True, transform=ToTensor())
    train_indices, val_indices = split_indices(len(dataset), 0.2)

    batch_size = 100

    # training sampler and data loader
    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)

    # validation sampler and data loader
    val_sampler = SubsetRandomSampler(val_indices)
    valid_dl = DataLoader(dataset, batch_size, sampler=val_sampler)

    return train_dl , valid_dl


if __name__ == "__main__":
    main()




