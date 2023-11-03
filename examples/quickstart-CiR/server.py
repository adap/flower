from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from models import Generator, Enclassifier
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Enclassifier().to(DEVICE)
net_gen = Generator().to(DEVICE)
n1 = [val.cpu().numpy() for _, val in net.state_dict().items()]
n2 = [val.cpu().numpy() for _, val in net_gen.state_dict().items()]
initial_params = ndarrays_to_parameters(n1)
initial_generator_params = ndarrays_to_parameters(n2)
all_labels = torch.arange(10).to(DEVICE)
one_hot_all_labels = torch.eye(10, dtype=torch.float).to(DEVICE)
z_g, mu_g, log_var_g = net_gen(one_hot_all_labels)
serialized_gen_stats = ndarrays_to_parameters(
    [mu_g.cpu().detach().numpy(), log_var_g.cpu().detach().numpy()]
)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedCiR(
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=initial_params,
    initial_generator_params=initial_generator_params,
    gen_stats=serialized_gen_stats,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
