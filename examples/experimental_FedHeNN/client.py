from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn.functional as F
from model_mnist import Net0, Net1, Net2, Net3
from dataset import load_mnist_data_partition
import argparse

warnings.filterwarnings("ignore", category=UserWarning)


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################


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
            loss3 = loss1

            loss3.backward()
            optimizer.step()

            break
        break


def test(net, device, testloader):
    """Validate the network on the entire test set."""
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = net(images)
            loss += F.nll_loss(outputs, labels, reduction="sum").item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            break

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main(part_idx):
    """Create model, load data, define Flower client, start Flower client."""

    # Load data
    trainloader, testloader, num_examples = load_mnist_data_partition(
        batch_size=32, partitions=5, RAD=False, subsample_RAD=True, use_cuda=False
    )[part_idx]
    # Model selection
    model_type = [Net0, Net1, Net2, Net3][part_idx]

    class VirtualClient(fl.client.NumPyClient):
        def __init__(self):
            # instantiate model
            self.net = model_type()
            # determine device
            self.device = torch.device("cpu")
            # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def set_parameters(self, parameters):
            for param in parameters:
                print(len(param), param.shape)
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            optimizer = torch.optim.Adam(self.net.parameters())

            # send model to device
            self.net.to(self.device)
            train(
                self.net,
                self.device,
                trainloader,
                optimizer,
                config,
            )
            return (
                self.get_parameters(),
                num_examples["trainset"],
                {},
            )

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)

            self.net.to(self.device)
            loss, accuracy = test(self.net, self.device, testloader)
            return (
                float(loss),
                num_examples["testset"],
                {"accuracy": float(accuracy)},
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
