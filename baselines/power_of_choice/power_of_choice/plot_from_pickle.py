import argparse
from logging import INFO
import os
import pickle
import matplotlib.pyplot as plt
from flwr.common.logger import log
from utils import plot_dloss_from_history

def main():
    parser = argparse.ArgumentParser(description="Plot Distributed Losses from History")
    parser.add_argument("pickle_path", type=str, help="Path to the pickle file containing history data")
    args = parser.parse_args()

    # Load metrics data from the pickle file
    with open(args.pickle_path, "rb") as pkl_file:
        history_data = pickle.load(pkl_file)

    log(
        INFO,
        f"Loaded history data {history_data}",
    )

    # Compute path to save the plot to by removing filename from the path
    save_plot_path = os.path.dirname(args.pickle_path)

    log(INFO, f"Saving plot to {save_plot_path}")

    # Plot distributed losses using the provided function
    plot_dloss_from_history(history_data["history"], save_plot_path)

if __name__ == "__main__":
    main()
