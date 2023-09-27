"""Plot a pickle file containing history of simulation."""

import argparse
import os
import pickle
from logging import INFO

from flwr.common.logger import log

from power_of_choice.utils import (
    plot_metrics_from_histories,
    plot_variances_training_loss_from_history,
)


def plot_multiplot(metrics_type, paths):
    """Plot a group of metrics in one figure."""
    num_plots = len(paths)
    titles = []
    histories = []

    for i in range(num_plots):
        title = input(f"Enter title for plot {i + 1}/{num_plots}: ")
        titles.append(title)

    for path, title in enumerate(zip(paths, titles)):
        with open(path, "rb") as pkl_file:
            history_data = pickle.load(pkl_file)

        log(
            INFO,
            f"Loaded history data {history_data}",
        )

        histories.append((title, history_data["history"]))

    # Compute path to save the plot to by removing filename from the last path given
    save_plot_path = os.path.dirname(path)

    log(INFO, f"Saving plot to {save_plot_path}")

    if metrics_type in ["paper_metrics"]:
        # Plot distributed losses using the provided function
        plot_metrics_from_histories(histories, save_plot_path)
    else:
        plot_variances_training_loss_from_history(histories, save_plot_path)


def main():
    """Plot a group of metrics in one figure."""
    parser = argparse.ArgumentParser(description="Plot Distributed Losses from History")
    parser.add_argument(
        "--metrics-type",
        type=str,
        choices=["paper_metrics", "variance"],
        help="Type of metrics to plot",
    )
    parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="Paths to the pickle files containing history data",
    )
    args = parser.parse_args()

    plot_multiplot(args.metrics_type, args.paths)


if __name__ == "__main__":
    main()
