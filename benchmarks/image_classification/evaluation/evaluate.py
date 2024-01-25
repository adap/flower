import argparse
import yaml
from logging import INFO

import torch
from torch.utils.data import DataLoader

from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from ..baseline.model import Net
from ..baseline.utils import apply_transforms


# Arguments parser
parser = argparse.ArgumentParser(description="Benchmark model evaluation")
parser.add_argument("--model-path", type=str, default="", help="Path of federated pre-trained model.")
parser.add_argument(
        "--mode",
        default="centralised",
        type=str,
        choices=["cen", "per"],
        help="Choices of centralised (cen) or personalised (per) evaluation.",
    )


# Borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * acc for num_examples, acc in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    weighted_acc = sum(accuracies) / sum(examples)

    return weighted_acc


def main():
    args = parser.parse_args()

    # Load static configs
    with open("../static/static_config.yaml", "r") as f:
        static_config = yaml.load(f, Loader=yaml.FullLoader)

    # Construct Federated Dataset
    partitioner = IidPartitioner(num_partitions=static_config["num_clients"])
    cifar10_fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init model
    model = Net()
    params_init = torch.load(args.model_path)
    model.load_state_dict(params_init, strict=True)
    model.to(device)

    # Centralised evaluation
    if args.mode == "cen":
        # Load centralised test set
        centralised_testset = cifar10_fds.load_full("test")

        # Apply transform to dataset
        testset = centralised_testset.with_transform(apply_transforms)

        # Construct dataloader
        testloader = DataLoader(testset, batch_size=50)

        # Evaluate
        loss, accuracy = test(model, testloader, device=device)

    # Personalised evaluation
    elif args.mode == "per":
        output_list = []
        for node_id in range(static_config["num_clients"]):
            # Get the partition corresponding to the i-th client
            client_dataset = cifar10_fds.load_partition(node_id, "train")

            # Client train/test split
            client_dataset_splits = client_dataset.train_test_split(test_size=static_config["test_size"], seed=static_config["seed"])
            testset = client_dataset_splits["test"]

            # Now we apply the transform to each batch
            testset = testset.with_transform(apply_transforms)

            # Construct dataloader
            testloader = DataLoader(testset, batch_size=64)

            # Evaluate
            loss, acc = test(model, testloader, device=device)
            output_list.append((len(testset), float(acc)))

        # Weighted average the accuracy values from clients
        accuracy = weighted_average(output_list)

    # Report the final accuracy
    log(INFO, f"Test accuracy: {accuracy}")
