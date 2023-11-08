from typing import List, Tuple
import flwr as fl
import torch
import warnings
import warnings
from collections import OrderedDict
from flwr.common import (
    Metrics,
    ndarrays_to_parameters,
    bytes_to_ndarray,
)
from tqdm import tqdm
from models import AlexNet, Generator
from utils_pacs import make_dataloaders
import torch.multiprocessing as mp


warnings.filterwarnings("ignore", category=UserWarning)
num_classes = 7
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_clients = 4
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:660"


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


net = AlexNet(num_classes=num_classes, latent_dim=4096, other_dim=1000).to(DEVICE)
net_gen = Generator(num_classes=num_classes, latent_dim=4096, other_dim=1000).to(DEVICE)
n1 = [val.cpu().numpy() for _, val in net.state_dict().items()]
n2 = [val.cpu().numpy() for _, val in net_gen.state_dict().items()]
initial_params = ndarrays_to_parameters(n1)
initial_generator_params = ndarrays_to_parameters(n2)
all_labels = torch.arange(num_classes).to(DEVICE)
one_hot_all_labels = torch.eye(num_classes, dtype=torch.float).to(DEVICE)
z_g, mu_g, log_var_g = net_gen(one_hot_all_labels)
serialized_gen_stats = ndarrays_to_parameters(
    [
        z_g.cpu().detach().numpy(),
        mu_g.cpu().detach().numpy(),
        log_var_g.cpu().detach().numpy(),
    ]
)


# Start Flower server
def server_task(serialized_gen_stats, initial_params, initial_generator_params):
    # Define strategy
    strategy = fl.server.strategy.FedCiR(
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_params,
        initial_generator_params=initial_generator_params,
        gen_stats=serialized_gen_stats,
        num_classes=num_classes,
        min_fit_clients=4,
        min_available_clients=4,
        min_evaluate_clients=4,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


def train(net1, trainloader, config, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
    lambda_reg = 0.5
    lambda_align = 5e-6
    all_labels = torch.arange(num_classes).to(DEVICE)

    z_g, mu_g, log_var_g = (
        torch.tensor(bytes_to_ndarray(config["z_g"])).to(DEVICE),
        torch.tensor(bytes_to_ndarray(config["mu_g"])).to(DEVICE),
        torch.tensor(bytes_to_ndarray(config["log_var_g"])).to(DEVICE),
    )
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred, mu, log_var = net1(images)
            # FL Loss
            loss_fl = criterion(pred, labels)

            # Reg Loss
            loss_reg = criterion(net1.clf(z_g), all_labels)

            # KL Div
            loss_align = 0.5 * (log_var_g[labels] - log_var - 1) + (
                log_var.exp() + (mu - mu_g[labels]).pow(2)
            ) / (2 * log_var_g[labels].exp())
            loss_align_reduced = loss_align.mean(dim=1).mean()
            loss = loss_fl + lambda_reg * loss_reg + lambda_align * loss_align_reduced
            loss.backward(retain_graph=True)
            optimizer.step()


def test(net1, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net1(images.to(DEVICE))[0]
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_idx) -> None:
        super().__init__()

        self.net1 = AlexNet(
            num_classes=num_classes, latent_dim=4096, other_dim=1000
        ).to(DEVICE)
        (loader_train, loader_test, _) = make_dataloaders(batch_size=64)
        self.trainloader = loader_train[client_idx]
        self.testloader = loader_test[client_idx]

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net1.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict1 = zip(self.net1.state_dict().keys(), parameters)
        state_dict1 = OrderedDict({k: torch.tensor(v) for k, v in params_dict1})
        self.net1.load_state_dict(state_dict1, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        train(self.net1, self.trainloader, config, epochs=5)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net1, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_task(client_idx):
    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_idx).to_client(),
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("spawned")
    except RuntimeError:
        pass
    processes = []

    server_process = mp.Process(
        target=server_task,
        args=(serialized_gen_stats, initial_params, initial_generator_params),
        name="Process-server",
    )
    server_process.start()
    processes.append(server_process)
    print(f"Started server")

    for cid in range(4):
        p = mp.Process(
            target=client_task,
            args=(cid,),
            name=f"Process-{cid}",
        )
        p.start()
        processes.append(p)
        print(f"Started {p.name}")

    # Wait for all processes to finish
    for p in processes:
        p.join()
        print(f"Finished {p.name}")
