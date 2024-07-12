"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from diskcache import Index


def _plot_line_plots(df):
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

        subset = df[(df["Model"] == model) & (df["Dataset"] == dataset)]

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
    logging.info(
        "Fig 10: FEDDEBUG performance at neuron activation threshold on 30 clients, "
        f"including five faulty clients. Figure generate {fname}"
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
        d = {}
        d["Faulty Clients"] = len(cfg.faulty_clients_ids)
        d["Total Clients"] = cfg.num_clients
        d["Architecture"] = cfg.model.name
        d["Dataset"] = cfg.dataset.name
        d["Accuracy"] = exp_dict["round2debug_result"][0]["eval_metrics"]["accuracy"]
        d["Epochs"] = cfg.client.epochs
        d["Noise Rate"] = cfg.noise_rate
        d["faulty_clients_ids"] = cfg.faulty_clients_ids
        d["Data Distribution"] = cfg.data_dist.dist_type
        d["Experiment Configuration Key"] = k

        all_df_rows.append(d)

    df = pd.DataFrame(all_df_rows)
    print(df)
    csv_name = "fed_debug_results.csv"
    df.to_csv(csv_name)
    logging.info(
        (
            "Table II: FEDDEBUG’s fault localization in 32 FL"
            "configurations with multiple faulty clients, ranging from two to seven."
            f"Results are also stored in {csv_name}"
        )
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

    df = pd.DataFrame(all_dicts)
    csv_name = "neuron_activation_threshold_variation.csv"

    print(df)
    df.to_csv(csv_name)
    _plot_line_plots(df)
