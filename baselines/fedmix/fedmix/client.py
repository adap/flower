"""..."""

import random
from collections import OrderedDict

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from flwr.client import NumPyClient
from hydra.utils import instantiate


class FedMixClient(NumPyClient):
    """..."""

    def __init__(self, net, trainloader, device, num_epochs, learning_rate):
        """..."""
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config):
        """..."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """..."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """..."""
        self.set_parameters(parameters)
        self.mashed_data = config["mashed_data"]
        self.mixup_ratio = config["mixup_ratio"]
        self.lr_decay_accumulated = config["lr_decay_accumulated"]
        self.local_update()
        return self.get_parameters({}), len(self.trainloader), {}

    def local_update(self):
        """..."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate * self.lr_decay_accumulated
        )
        self.net.train()
        for _ in range(self.num_epochs):
            self.training_loop(criterion, optimizer)

    def training_loop(self, criterion, optimizer):
        """..."""
        for images, labels in self.trainloader:
            mashed_image, mashed_label = random.choice(self.mashed_data)
            images, labels = images.to(self.device), labels.to(self.device)
            mashed_image, mashed_label = mashed_image[None, :].to(
                self.device
            ), mashed_label[None, :].to(self.device)

            num_classes = len(mashed_label[0])
            mashed_labels = mashed_label.expand_as(F.one_hot(labels, num_classes))

            optimizer.zero_grad()

            scaled_images = (1 - self.mixup_ratio) * images
            scaled_images.requires_grad_()

            intermediate_output = self.net(scaled_images)

            l1 = (1 - self.mixup_ratio) * criterion(intermediate_output, labels)
            l2 = self.mixup_ratio * criterion(intermediate_output, mashed_labels)

            gradients = autograd.grad(
                outputs=l1, inputs=scaled_images, create_graph=True, retain_graph=True
            )[0]

            l3 = self.mixup_ratio * torch.inner(
                gradients.flatten(start_dim=1), mashed_image.flatten(start_dim=1)
            )
            l3 = torch.mean(l3)

            loss = l1 + l2 + l3
            loss.backward()
            optimizer.step()

    def evaluate(self, parameters, config):
        """..."""
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.trainloader), {"accuracy": float(accuracy)}

    def test(self):
        """..."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.net.eval()
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if len(self.trainloader.dataset) == 0:
            raise ValueError("Testloader can't be 0, exiting...")
        loss /= len(self.trainloader.dataset)
        accuracy = correct / total
        return loss, accuracy


def gen_client_fn(config, trainloaders, model):
    """..."""

    def client_fn(cid):
        """..."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        trainloader = trainloaders[int(cid)]

        return FedMixClient(
            net, trainloader, device, config.num_local_epochs, config.lr
        )

    return client_fn
