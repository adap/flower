"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torchvision import transforms
from logging import INFO
from flwr.common.logger import log


def seed_everything(seed=786):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_localization_accuracy(true_faulty_clients, predicted_faulty_clients):
    """Calculate the fault localization accuracy."""
    true_preds = 0
    total = 0
    for client, predicted_faulty_count in predicted_faulty_clients.items():
        if client in true_faulty_clients:
            true_preds += predicted_faulty_count

        total += predicted_faulty_count

    accuracy = (true_preds / total) * 100
    return accuracy


def create_transform():
    """Creates and returns the transformation pipeline."""

    tfms = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return tfms


def set_exp_key(cfg):
    """Set the experiment key."""
    key = (
        f"{cfg.model}-{cfg.dataset.name}-"
        f"faulty_clients[{cfg.malicious_clients_ids}]-"
        f"noise_rate{cfg.noise_rate}-"
        f"TClients{cfg.num_clients}-"
        f"-clientsPerR{cfg.clients_per_round})"
        f"-{cfg.distribution}"
        f"-batch{cfg.client.batch_size}-epochs{cfg.client.epochs}-"
        f"lr{cfg.client.lr}"
    )
    return key

def config_sim_resources(cfg):
    """Configure the resources for the simulation."""
    client_resources = {"num_cpus": cfg.client_resources.num_cpus}
    if cfg.device == "cuda":
        client_resources["num_gpus"] = cfg.client_resources.num_gpus

    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": cfg.total_gpus}
    backend_config = {
        "client_resources": client_resources,
        "init_args": init_args,
    }
    return backend_config


def get_parameters(model):
    """Return model parameters as a list of NumPy ndarrays."""
    model = model.cpu()
    return [val.cpu().detach().clone().numpy() for _, val in model.state_dict().items()]


def set_parameters(net, parameters):
    """Set model parameters from a list of NumPy ndarrays."""
    net = net.cpu()
    params_dict = zip(net.state_dict().keys(), parameters)
    new_state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    net.load_state_dict(new_state_dict, strict=True)


def plot_metrics(gm_accs, feddebug_accs, cfg):
    """Plot the metrics with legend and save the plot."""
    fig, ax = plt.subplots(
        figsize=(5, 4))  # Increase figure size for better readability

    # Convert accuracy to percentages
    gm_accs = [x * 100 for x in gm_accs][1:]

    # Plot lines with distinct styles for better differentiation
    ax.plot(gm_accs, label="Global Model", linestyle='-', linewidth=2)
    ax.plot(feddebug_accs, label="FedDebug", linestyle='--', linewidth=2)

    # Set labels with font settings
    ax.set_xlabel("Training Round", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')

    # Set title with font settings
    title = f"{cfg.distribution}-{cfg.model}-{cfg.dataset.name}"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Customize grid for better readability
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # Improve tick parameters
    ax.xaxis.set_major_locator(ticker.MaxNLocator(
        integer=True))  # Show integer values for rounds
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set legend with better positioning and font size
    ax.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, shadow=True)

    # Tight layout to avoid clipping
    fig.tight_layout()

    # Save the figure with a higher resolution for publication quality
    fname = f"{title}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    log(INFO, f"Saved plot at {fname}")
