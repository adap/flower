import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List
import ray
from torch.utils.data import DataLoader

import torch

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import configure
from flwr.common.typing import Scalar
from strategy.FedGuideVisualise import VisualiseFedGuide
from utils.utils_mnist import (
    load_data_mnist,
    train,
    test,
    visualize_gen_image,
    visualize_gmm_latent_representation,
    non_iid_train_iid_test,
    iid_train_iid_test,
    alignment_dataloader,
    eval_reconstrution,
    visualize_plotly_latent_representation,
    sample_latents,
    get_fisher_ratio,
    train_single_loader,
    VAE,
    train_alternate_frozen,
    VAE_CNN,
)
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import ray

NUM_CLIENTS = 5
NUM_CLASSES = 10
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=5,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=1 / 3,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument(
    "--num_rounds", "-r", type=int, default=50, help="Number of FL rounds."
)
parser.add_argument(
    "--identifier", "-i", type=str, required=True, help="Name of experiment."
)
parser.add_argument(
    "--latent_dim", "-dim", type=int, required=True, help="Latent dimension."
)

import wandb

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LATENT_DIM = args.latent_dim
IDENTIFIER = args.identifier
if not os.path.exists(IDENTIFIER):
    os.makedirs(IDENTIFIER)

configure(identifier=IDENTIFIER, filename=f"logs_{IDENTIFIER}.log")
# TODO:hardcoded
# ALIGNMENT_DATA = alignment_dataloader(
#     samples_per_class=100, batch_size=100 * NUM_CLASSES, only_data=True
# )


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, align_loader, cid):
        self.trainset = trainset
        self.valset = valset
        self.align_loader = align_loader
        self.cid = cid

        # Instantiate model
        self.model = VAE_CNN(z_dim=LATENT_DIM)

        # Determine device
        self.device = DEVICE
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        if config["local_param_dict"] is not None:

            set_params(
                self.model,
                parameters_to_ndarrays(
                    config["local_param_dict"][f"client_param_{self.cid}"]
                ),
            )
        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader
        g = torch.Generator()
        g.manual_seed(self.cid)
        trainloader = DataLoader(self.trainset, batch_size=batch, generator=g)
        # trainloader = DataLoader(
        #     ALIGNMENT_DATA, batch_size=100 * NUM_CLASSES, generator=g
        # )

        # Define optimizer
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Train
        # loss = train(
        #     self.model,
        #     trainloader,
        #     optimizer,
        #     config,
        #     epochs=epochs,
        #     device=self.device,
        #     num_classes=NUM_CLASSES,
        # )
        local_loss_term, loss_ref_term, align_term, latent_diff_term = (
            train_alternate_frozen(
                self.model,
                trainloader,
                self.align_loader,
                config,
                epochs=epochs,
                device=self.device,
            )
        )

        true_img, gen_img = visualize_gen_image(
            self.model,
            DataLoader(self.valset, batch_size=64, shuffle=True),
            self.device,
            f'for_client_{self.cid}_train_at_round_{config.get("server_round")}',
            folder=IDENTIFIER,
        )
        latent_rep = visualize_plotly_latent_representation(
            self.model,
            DataLoader(self.valset, batch_size=64),
            self.device,
            num_class=NUM_CLASSES,
            use_PCA=True,
        )
        # Return local model and statistics
        return (
            self.get_parameters({}),
            len(trainloader.dataset),  # TODO: change to align_loader
            {
                "cid": self.cid,
                "local_loss_term": local_loss_term,
                "loss_ref_term": loss_ref_term,
                "align_term": align_term,
                "latent_diff_term": latent_diff_term,
                "true_image": true_img,
                "gen_image": gen_img,
                "latent_rep": latent_rep,
                "client_round": config["server_round"],
                "total_loss": local_loss_term + loss_ref_term,
            },
        )

    def evaluate(self, parameters, config):
        if config["local_param_dict"] is not None:
            client_params = parameters_to_ndarrays(
                config["local_param_dict"][f"client_param_{self.cid}"]
            )
            model_params = [
                val.cpu().numpy() for _, val in self.model.state_dict().items()
            ]
            print(f"client_params{self.cid}: {client_params[7]}")
            print(f"model_params {self.cid}: {model_params[7]}")
            print(f"get_params {self.cid}: {self.get_parameters({})[7]}")
            set_params(
                self.model,
                parameters_to_ndarrays(
                    config["local_param_dict"][f"client_param_{self.cid}"]
                ),
            )

        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device, cnn=True)
        fisher_score = get_fisher_ratio(self.model, valloader, LATENT_DIM, self.device)

        # Return statistics
        return (
            float(loss),
            len(valloader.dataset),
            {
                "accuracy": float(accuracy),
                "cid": self.cid,
                "local_val_loss": float(loss),
                "fisher_score": float(fisher_score),
            },
        )


def get_client_fn(train_partitions, val_partitions, align_loader):
    """Return a function to construct a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        trainset, valset = train_partitions[int(cid)], val_partitions[int(cid)]

        # Create and return client
        return FlowerClient(trainset, valset, align_loader, int(cid)).to_client()

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
    # accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    weighted_metrics = {
        key: [num_examples * m[key] for num_examples, m in metrics if key != "cid"]
        for num_examples, m in metrics
        for key in m
        if key != "cid"
    }
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    # return {"accuracy": sum(accuracies) / sum(examples)}
    return {key: sum(value) / sum(examples) for key, value in weighted_metrics.items()}


def main():
    # Parse input arguments
    run = wandb.init(
        entity="mak",
        group="guide50",
        reinit=True,
    )

    print(f"running these hparams-> {wandb.config}")
    wandb.define_metric("server_round")
    wandb.define_metric("global_*", step_metric="server_round")
    wandb.define_metric("generated_*", step_metric="server_round")
    wandb.define_metric("client_round")
    wandb.define_metric("train_*", step_metric="client_round")
    wandb.define_metric("eval_*", step_metric="client_round")

    samples_per_class = wandb.config["sample_per_class"]

    ALIGNMENT_DATALOADER = alignment_dataloader(
        samples_per_class=samples_per_class,
        batch_size=samples_per_class * NUM_CLASSES,
    )

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": wandb.config["epochs"],  # Number of local epochs done by clients
            "batch_size": wandb.config["batch_size"],
            "server_round": server_round,
            "beta": wandb.config["beta"],
            "sample_per_class": wandb.config["sample_per_class"],
            "lambda_reg": wandb.config["lambda_reg"],
            "lambda_align": wandb.config["lambda_align"],
            "lambda_align2": wandb.config["lambda_align2"],
            "lambda_reg_dec": wandb.config["lambda_reg_dec"],
            "lambda_latent_diff": wandb.config["lambda_latent_diff"],
            "lr_g": wandb.config["lr_g"],
            "steps_g": wandb.config["steps_g"],
            "latent_dim": LATENT_DIM,
            "cnn": True,  # for VAE_CNN
        }
        return config

    def eval_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "server_round": server_round,
        }
        return config

    def get_evaluate_fn(testset):
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
        ):
            """Use the entire test set for evaluation."""

            # Determine device
            device = DEVICE

            model = VAE_CNN(z_dim=LATENT_DIM)
            model.to(device)
            set_params(model, parameters)
            model.eval()
            if server_round == 0 or server_round == args.num_rounds:
                with open(
                    f"{IDENTIFIER}/{run.name}-weights_gen_round_{server_round}.npy",
                    "wb",
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
            latent_reps = visualize_plotly_latent_representation(
                model,
                testloader,
                device,
                num_class=NUM_CLASSES,
                use_PCA=True,
            )
            global_val_loss = eval_reconstrution(model, testloader, device, cnn=True)
            with torch.no_grad():
                z_sample_np = sample_latents(model, testloader, device, 64)
                if z_sample_np is None:
                    return
                z_sample = torch.tensor(z_sample_np, dtype=torch.float32).to(device)
                recon = model.decoder(z_sample).cpu()
                recon = recon.view(-1, 1, 28, 28)
                fig, ax = plt.subplots(figsize=(10, 10))
                img = make_grid(recon, nrow=8, normalize=True).permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis("off")
                fisher_score = float(
                    get_fisher_ratio(model, testloader, LATENT_DIM, device)
                )
            decouple_config = {k: wandb.Image(v) for k, v in config.items()}
            wandb.log(
                {
                    f"global_true_image": wandb.Image(true_img),
                    f"global_gen_image": wandb.Image(gen_img),
                    f"global_latent_rep": latent_reps,
                    f"global_val_loss": global_val_loss,
                    f"global_fisher_score": fisher_score,
                    f"server_round": server_round,
                    f"generated_gen_round": plt,
                    **decouple_config,
                },
                step=server_round,
            )
            plt.close("all")

        return evaluate

    # Download dataset and partition it
    trainsets, valsets = non_iid_train_iid_test()
    net = VAE_CNN(z_dim=LATENT_DIM).to(DEVICE)

    n1 = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_params = ndarrays_to_parameters(n1)

    gen_net = VAE_CNN(z_dim=LATENT_DIM, encoder_only=True).to(DEVICE)
    n2 = [val.cpu().numpy() for _, val in gen_net.state_dict().items()]
    initial_gen_params = ndarrays_to_parameters(n2)

    strategy = VisualiseFedGuide(
        initial_parameters=initial_params,
        initial_generator_params=initial_gen_params,
        gen_model=gen_net,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(valsets[-1]),  # Global evaluation function
        alignment_dataloader=ALIGNMENT_DATALOADER,
        lr_g=wandb.config["lr_g"],
        steps_g=wandb.config["steps_g"],
        lambda_align_g=wandb.config["lambda_align_g"],
        device=DEVICE,
        num_classes=NUM_CLASSES,
        latent_dim=LATENT_DIM,
        global_eval_data=valsets[-1],
        folder_name=IDENTIFIER,
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(trainsets, valsets, ALIGNMENT_DATALOADER),
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
            "epochs": {"values": [10]},
            "batch_size": {"values": [128]},
            "beta": {"values": [0]},
            "sample_per_class": {"values": [100]},
            "lr_g": {"values": [1e-3]},
            "steps_g": {"values": [100]},
            "lambda_reg": {"values": [1]},
            "lambda_align_g": {"values": [1]},  # for generator KL lambda
            "lambda_align": {"values": [0]},  # cka
            "lambda_align2": {"values": [0]},
            "lambda_reg_dec": {"values": [0, 0.1]},
            "lambda_latent_diff": {"values": [0]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project=IDENTIFIER)

    wandb.agent(sweep_id, function=main, count=2)
