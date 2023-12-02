import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader
from logging import WARNING, DEBUG
import torch

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.common.logger import configure, log
from flwr.common.typing import Scalar
import wandb
import os

os.environ["WANDB_START_METHOD"] = "thread"

from utils_mnist import (
    test,
    visualize_gen_image,
    visualize_gmm_latent_representation,
    non_iid_train_iid_test_6789,
    subset_alignment_dataloader,
    train_align,
    eval_reconstrution,
)
from utils_mnist import VAE
import os
import numpy as np
import matplotlib.pyplot as plt

NUM_CLIENTS = 2
NUM_CLASSES = 4
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=6,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=1 / 2,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument("--num_rounds", type=int, default=20, help="Number of FL rounds.")
parser.add_argument("--identifier", type=str, required=True, help="Name of experiment.")
args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IDENTIFIER = args.identifier
if not os.path.exists(IDENTIFIER):
    os.makedirs(IDENTIFIER)
configure(identifier=IDENTIFIER, filename=f"logs_{IDENTIFIER}.log")
import ray


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, cid):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid

        # Instantiate model
        self.model = VAE()

        # Determine device
        self.device = DEVICE
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Train
        vae_loss_term, reg_term, align_term = train_align(
            self.model,
            trainloader,
            optimizer,
            config,
            epochs=epochs,
            device=self.device,
            num_classes=NUM_CLASSES,
        )

        true_img, gen_img = true_img, gen_img = visualize_gen_image(
            self.model,
            DataLoader(self.valset, batch_size=64, shuffle=True),
            self.device,
            f'for_client_{self.cid}_train_at_round_{config.get("server_round")}',
            folder=IDENTIFIER,
        )
        latent_reps = visualize_gmm_latent_representation(
            self.model,
            DataLoader(self.valset, batch_size=64),
            self.device,
            f'for_client_{self.cid}_train_at_round_{config.get("server_round")}',
            folder=IDENTIFIER,
            num_class=NUM_CLASSES,
        )

        # Return local model and statistics
        return (
            self.get_parameters({}),
            len(trainloader.dataset),
            {
                "cid": self.cid,
                "vae_loss_term": vae_loss_term,
                "reg_term": reg_term,
                "align_term": align_term,
                "true_image": true_img,
                "gen_image": gen_img,
                "latent_rep": latent_reps,
                "client_round": config["server_round"],
            },
        )

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return statistics
        return (
            float(loss),
            len(valloader.dataset),
            {
                "accuracy": float(accuracy),
                "cid": self.cid,
                "local_val_loss": float(loss),
            },
        )


def get_client_fn(train_partitions, val_partitions):
    """Return a function to construct a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        trainset, valset = train_partitions[int(cid)], val_partitions[int(cid)]

        # Create and return client
        return FlowerClient(trainset, valset, cid).to_client()

    return client_fn


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    run = wandb.init(
        entity="mak",
        group="cir",
        reinit=True,
    )

    print(f"running these hparams-> {wandb.config}")
    wandb.define_metric("server_round")
    wandb.define_metric("global_*", step_metric="server_round")
    wandb.define_metric("client_round")
    wandb.define_metric("train_*", step_metric="client_round")
    wandb.define_metric("eval_*", step_metric="client_round")

    # Parse input arguments
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": wandb.config["epochs"],  # Number of local epochs done by clients
            "batch_size": wandb.config["batch_size"],
            "server_round": server_round,
            "sample_per_class": wandb.config["sample_per_class"],
            "lambda_reg": wandb.config["lambda_reg"],
            "lambda_align": wandb.config["lambda_align"],
            "lr_g": wandb.config["lr_g"],
            "steps_g": wandb.config["steps_g"],
        }
        return config

    def get_evaluate_fn(
        testset,
    ):
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
        ):
            """Use the entire test set for evaluation."""
            # Determine device
            device = DEVICE

            model = VAE()
            model.to(device)
            set_params(model, parameters)
            if server_round == 0 or server_round == args.num_rounds:
                with open(
                    f"{IDENTIFIER}/weights_cir_round_{server_round}.npy", "wb"
                ) as f:
                    np.save(f, np.array(parameters, dtype=object))
                wandb.watch(model)

            testloader = DataLoader(testset, batch_size=64, shuffle=True)
            true_img, gen_img = visualize_gen_image(
                model,
                testloader,
                device,
                f"server_eval_{server_round}",
                folder=IDENTIFIER,
            )
            latent_reps = visualize_gmm_latent_representation(
                model,
                testloader,
                device,
                f"server_eval_{server_round}",
                folder=IDENTIFIER,
                num_class=NUM_CLASSES,
            )
            global_val_loss = eval_reconstrution(model, testloader, device)
            wandb.log(
                {
                    f"global_true_image": wandb.Image(true_img),
                    f"global_gen_image": wandb.Image(gen_img),
                    f"global_latent_rep": wandb.Image(latent_reps),
                    f"global_val_loss": global_val_loss,
                    "server_round": server_round,
                }
            )
            plt.close("all")

        return evaluate

    # Download dataset and partition it
    trainsets, valsets = non_iid_train_iid_test_6789()
    net = VAE().to(DEVICE)
    gen_net = VAE(encoder_only=True).to(DEVICE)
    n1 = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_params = ndarrays_to_parameters(n1)
    n2 = [val.cpu().numpy() for _, val in gen_net.state_dict().items()]
    initial_gen_params = ndarrays_to_parameters(n2)
    samples_per_class = wandb.config["sample_per_class"]
    strategy = fl.server.strategy.FedCiR(
        initial_parameters=initial_params,
        initial_generator_params=initial_gen_params,
        gen_stats=initial_gen_params,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(valsets[-1]),  # Global evaluation function
        alignment_dataloader=subset_alignment_dataloader(
            samples_per_class=samples_per_class,
            batch_size=samples_per_class * NUM_CLASSES,
        ),
        lr_g=wandb.config["lr_g"],
        steps_g=wandb.config["steps_g"],
        device=DEVICE,
        num_classes=NUM_CLASSES,
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(trainsets, valsets),
        num_clients=NUM_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args={
            "include_dashboard": True,  # we need this one for tracking
        },
    )
    ray.shutdown()
    wandb.finish()


if __name__ == "__main__":
    sweep_config = {
        "method": "random",
        "metric": {"name": "global_val_loss", "goal": "minimize"},
        "parameters": {
            "sample_per_class": {"values": [50, 100, 150, 200, 250, 300]},
            "lambda_reg": {"min": 0.0, "max": 1.0},
            "lambda_align": {"values": [1, 10, 50, 100]},
            "lr_g": {"values": [1e-3, 1e-2, 1e-4]},
            "steps_g": {"values": [5, 10, 15, 20]},
            "epochs": {"values": [1, 2, 3, 4, 5]},
            "batch_size": {"values": [32, 64, 128]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project=IDENTIFIER)

    wandb.agent(sweep_id, function=main, count=15)
