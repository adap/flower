import argparse
from collections import OrderedDict
from flamby.datasets.fed_heart_disease import NUM_CLIENTS

import flwr as fl
import torch

from client import Client
from strategy import NewtonRaphsonStrategy
from utils import Baseline, get_data, validate


def gen_eval_fun(cpu_only):
    net = Baseline()
    data_loader = get_data(0, train=False)

    use_gpu = torch.cuda.is_available() and not (cpu_only)

    def eval_fun(round, params, config):
        params_dict = zip(net.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        loss, metrics = validate(net, data_loader, use_gpu)

        return loss, {"metric": metrics}

    return eval_fun


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


def get_client_fn(cpu_only):
    def client_fn(cid):
        net = Baseline()
        train_data = get_data(int(cid), train=True)
        test_data = get_data(int(cid), train=False)
        return Client(net, train_data, test_data, cpu_only, batch_size=4)

    return client_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=3,
        help="Number of rounds to run FL for.",
    )
    parser.add_argument("--cpu_only", action="store_true")
    args = parser.parse_args()

    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(args.cpu_only),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.n_rounds),
        strategy=NewtonRaphsonStrategy(
            evaluate_fn=gen_eval_fun(args.cpu_only),
            evaluate_metrics_aggregation_fn=weighted_average,
            damping_factor=0.8,
        ),
        ray_init_args={"logging_level": "error", "log_to_driver": False},
    )
