"""This code takes the evalution results from the directory results/ and
creates a plot to compare e.g. fedbn with fedavg."""


import json

import matplotlib.pyplot as plt  # type: ignore

fedavg_step_number = []
fedavg_loss = []
fedbn_step_number = []
fedbn_loss = []


def get_evaluation_numbers() -> None:
    """Open the json files to get the evaluation results."""
    with open("results/SVHN_fedavg_results.json") as fedavg_file:
        fedavg_list = json.load(fedavg_file)
    with open("results/SVHN_fedbn_results.json") as fedbn_file:
        fedbn_list = json.load(fedbn_file)
    for item in fedavg_list:
        if "train_loss" in item:
            fedavg_step_number.append(item["fl_round"])
            fedavg_loss.append(item["train_loss"])
    for item in fedbn_list:
        if "train_loss" in item:
            fedbn_step_number.append(item["fl_round"])
            fedbn_loss.append(item["train_loss"])


def main() -> None:
    """Plot evaluation results."""
    get_evaluation_numbers()
    # pylint: disable= unused-variable, invalid-name
    fig, ax = plt.subplots()
    fedavg = ax.plot(fedavg_step_number, fedavg_loss, label="FedAvg")
    fedbn = ax.plot(fedbn_step_number, fedbn_loss, label="FedBN")
    ax.legend()
    plt.axis([-3, 100, -0.1, 2.5])
    plt.ylabel("Training loss")
    plt.xlabel("Number of FL round")
    plt.title("SVHN")
    plt.savefig("convergence_rate.png")


if __name__ == "__main__":
    main()
