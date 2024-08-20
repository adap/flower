"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from logging import INFO

import matplotlib.pyplot as plt
import pandas as pd
import torch
from diskcache import Index
from flwr.common.logger import log


def _plot_line_plots(df_plot):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        "Malicious Clients Localization Accuracy vs Neuron Activation Threshold"
    )

    # Define combinations of models and datasets
    combinations = [
        ("resnet50", "cifar10"),
        ("resnet50", "femnist"),
        ("densenet121", "cifar10"),
        ("densenet121", "femnist"),
    ]

    # Plot each combination
    for idx, (model, dataset) in enumerate(combinations):
        row = idx // 2
        col = idx % 2

        subset = df_plot[(df_plot["Model"] == model) & (df_plot["Dataset"] == dataset)]

        # Sort by threshold to ensure correct line plotting
        subset = subset.sort_values("Neuron Activation Threshold")

        axs[row, col].plot(
            subset["Neuron Activation Threshold"],
            subset["Malacious Client(s) Localization Accuracy (%)"],
            "o-",
            color="black",
        )

        axs[row, col].set_xlim(0, 1)
        axs[row, col].set_ylim(0, 110)
        axs[row, col].set_xlabel("Neuron Activation Threshold")
        axs[row, col].set_ylabel("Localization Accuracy (%)")
        axs[row, col].set_title(f"({chr(97+idx)}) {model} and {dataset.upper()}")

        # # Add more points to make the graph look closer to the image
        # x_interp = np.linspace(0, 1, 10)
        # y_interp = np.interp(x_interp, subset['Neuron Activation Threshold'],
        #                     subset['Malacious Client(s) Localization Accuracy (%)'])
        # axs[row, col].plot(x_interp, y_interp, 'o-', color='black', markersize=3)

    plt.tight_layout()
    plt.show()
    fname = "Figure-10.pdf"
    plt.savefig(fname)
    plt.savefig("Figure-10.png")
    log(
        INFO,
        "Fig 10: FEDDEBUG performance at neuron activation threshold on 30 clients, "
        f"including five faulty clients. Figure generate {fname}",
    )


def generate_table2_csv(cache_path, igonre_keys):
    """Generate Table II from the paper."""
    cache = Index(cache_path)
    all_exp_keys = cache.keys()
    all_df_rows = []

    for k in all_exp_keys:
        if k in igonre_keys:
            continue
        exp_dict = cache[k]
        cfg = exp_dict["debug_cfg"]
        table_d = {}
        table_d["Faulty Clients"] = len(cfg.faulty_clients_ids)
        table_d["Total Clients"] = cfg.num_clients
        table_d["Architecture"] = cfg.model.name
        table_d["Dataset"] = cfg.dataset.name
        table_d["Accuracy"] = exp_dict["round2debug_result"][0]["eval_metrics"][
            "accuracy"
        ]
        table_d["Epochs"] = cfg.client.epochs
        table_d["Noise Rate"] = cfg.noise_rate
        table_d["faulty_clients_ids"] = cfg.faulty_clients_ids
        table_d["Data Distribution"] = cfg.data_dist.dist_type
        table_d["Experiment Configuration Key"] = k

        all_df_rows.append(table_d)

    df_table = pd.DataFrame(all_df_rows)
    print(df_table)
    csv_name = "fed_debug_results.csv"
    df_table.to_csv(csv_name)
    log(
        INFO,
        (
            "Table II: FEDDEBUG’s fault localization in 32 FL"
            "configurations with multiple faulty clients, ranging from two to seven."
            f"Results are also stored in {csv_name}"
        ),
    )


def gen_thresholds_exp_graph(cache_path, threshold_exp_key):
    """Generate the graph for the neuron activation threshold experiment."""
    cache = Index(cache_path)
    exp_dict = cache[threshold_exp_key]

    all_keys = exp_dict.keys()

    all_dicts = []

    for k in all_keys:
        cfg = exp_dict[k]["cfg"]
        na2avg_acc = exp_dict[k]["na2acc"]
        temp_li = [
            {
                "Neuron Activation Threshold": na,
                "Malacious Client(s) Localization Accuracy (%)": acc,
                "Dataset": cfg.dataset.name,
                "Faulty Client (s)": cfg.faulty_clients_ids,
                "Model": cfg.model.name,
                "Total Clients": cfg.num_clients,
                "Epochs": cfg.client.epochs,
            }
            for na, acc in na2avg_acc.items()
        ]

        all_dicts += temp_li

    df_na_t = pd.DataFrame(all_dicts)
    csv_name = "neuron_activation_threshold_variation.csv"

    print(df_na_t)
    df_na_t.to_csv(csv_name)
    _plot_line_plots(df_na_t)


def set_exp_key(cfg):
    """Set the experiment key."""
    key = (
        f"{cfg.model.name}-{cfg.dataset.name}-"
        f"faulty_clients[{cfg.faulty_clients_ids}]-"
        f"noise_rate{cfg.noise_rate}-"
        f"TClients{cfg.data_dist.num_clients}-"
        f"{cfg.strategy.name}-(R{cfg.strategy.num_rounds}"
        f"-clientsPerR{cfg.strategy.clients_per_round})"
        f"-{cfg.data_dist.dist_type}{cfg.data_dist.dirichlet_alpha}"
        f"-batch{cfg.data_dist.batch_size}-epochs{cfg.client.epochs}-"
        f"lr{cfg.client.lr}"
    )
    return key


def config_sim_resources(cfg):
    """Configure the resources for the simulation."""
    client_resources = {"num_cpus": cfg.client_cpus}
    if cfg.device == "cuda":
        client_resources["num_gpus"] = cfg.client_gpu

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
