from collections import OrderedDict
import warnings
from flwr.common import (
    Scalar,
)
import flwr as fl
import torch
import torch.nn.functional as F
from model_mnist import Net0, Net1, Net2, Net3
from dataset import load_mnist_data_partition
import argparse
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
from similarity_utils import cka_torch, gram_linear_torch
from torch.autograd import Variable

Scalar = Union[bool, bytes, float, int, str]

warnings.filterwarnings("ignore", category=UserWarning)


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################


def proximal_term(cfg, local_model, device, train=True):
    delta_list = []
    global_prev_K_value = torch.tensor(
        np.asarray(json.loads(cfg["K_final"])), requires_grad=True, dtype=torch.float32
    )
    trainloader_RAD, testloader_RAD, num_examples_RAD = load_mnist_data_partition(
        batch_size=32,
        partitions=5,
        RAD=True,
        subsample_RAD=True,
        use_cuda=False,
        input_seed=int(cfg["epoch_global"]),
    )
    if train:
        dataloader_RAD = trainloader_RAD
    else:
        dataloader_RAD = testloader_RAD
    for images_RAD, labels_RAD in dataloader_RAD:
        images_RAD, labels_RAD = images_RAD.to(device), labels_RAD.to(device)
        intermediate_activation_local, _ = local_model(images_RAD)
        cka_from_examples = cka_torch(
            global_prev_K_value,
            gram_linear_torch(intermediate_activation_local),
        )
        delta_list.append(cka_from_examples)
    # print(gram_linear_torch(intermediate_activation_local).shape)
    # print(f"k_global:{global_prev_K_value.shape}")
    return Variable(torch.mean(torch.FloatTensor(delta_list)), requires_grad=True)


def train(net, device, trainloader, optimizer, cfg):
    """Train the network on the training set."""
    eta = (int(cfg["epoch_global"]) + 1) / (int(cfg["num_rounds"]) + 2)
    epochs = int(cfg["epochs"])
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            _, outputs = net(images)
            loss1 = F.nll_loss(outputs, labels) * (1 - eta)
            loss2 = proximal_term(cfg, net, device) * (eta)

            loss3 = loss1 + loss2

            loss3.backward()
            optimizer.step()

        #     break
        # break
    return torch.mean(loss1).item(), torch.mean(loss2).item()


def test(net, device, testloader, cfg):
    """Validate the network on the entire test set."""
    correct, total, loss, cka_score = 0, 0, 0.0, 0.0
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            _, outputs = net(images)
            loss += F.nll_loss(outputs, labels, reduction="sum").item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            cka_score += proximal_term(cfg, net, device, train=False)
            # break

    loss /= len(testloader.dataset)
    accuracy = correct / total
    cka_score /= idx + 1
    return loss, accuracy, cka_score


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main(part_idx):
    """Create model, load data, define Flower client, start Flower client."""

    # Load data for clients
    trainloader, testloader, num_examples = load_mnist_data_partition(
        batch_size=32, partitions=5, RAD=False, subsample_RAD=True, use_cuda=False
    )[part_idx]
    # Model selection
    tensor_type = list("abcd")[part_idx]

    class VirtualClient(fl.client.NumPyClient):
        def __init__(self):
            # determine device
            self.device = torch.device("cpu")
            # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.properties: Dict[str, Scalar] = {"tensor_type": f"model_{tensor_type}"}

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            if config["tensor_type"] == "model_a":
                self.net = Net0()
            if config["tensor_type"] == "model_b":
                self.net = Net1()
            if config["tensor_type"] == "model_c":
                self.net = Net2()
            if config["tensor_type"] == "model_d":
                self.net = Net3()

            # print(f"name of net:{str(self.net)[:4]}")
            self.set_parameters(parameters)
            optimizer = torch.optim.Adam(self.net.parameters())

            # send model to device
            self.net.to(self.device)
            loss1, loss2 = train(
                self.net,
                self.device,
                trainloader,
                optimizer,
                config,
            )
            return (
                self.get_parameters(),
                num_examples["trainset"],
                {"tensor_type": config["tensor_type"], "loss1": loss1, "loss2": loss2},
            )

        def evaluate(self, parameters, config):
            if config["tensor_type"] == "model_a":
                self.net = Net0()
            if config["tensor_type"] == "model_b":
                self.net = Net1()
            if config["tensor_type"] == "model_c":
                self.net = Net2()
            if config["tensor_type"] == "model_d":
                self.net = Net3()
            self.set_parameters(parameters)

            self.net.to(self.device)
            loss, accuracy, cka_score = test(self.net, self.device, testloader, config)
            return (
                float(loss),
                num_examples["testset"],
                {
                    "tensor_type": config["tensor_type"],
                    "accuracy": float(accuracy),
                    "cka_score": float(cka_score),
                    "loss": float(loss),
                },
            )

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=VirtualClient())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--part_idx", type=int, help="partition_id", required=True
    )

    args_partition = parser.parse_args()

    main(args_partition.part_idx)
