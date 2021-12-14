import json
import matplotlib.pyplot as plt
import numpy as np

fedavg_step_number = []
fedavg_loss = []
fedbn_step_number = []
fedbn_loss = []


def main() -> None:
    with open("SVHN_fedavg_results.json") as f:
        fedavg_list = json.load(f)
    with open("SVHN_fedbn_results.json") as f:
        fedbn_list = json.load(f)
    for item in fedavg_list:
        if "train_loss" in item:
            fedavg_step_number.append(item["fl_round"])
            fedavg_loss.append(item["train_loss"])
    for item in fedbn_list:
        if "train_loss" in item:
            fedbn_step_number.append(item["fl_round"])
            fedbn_loss.append(item["train_loss"])


if __name__ == "__main__":
    main()
    fig, ax = plt.subplots()
    fedavg = ax.plot(fedavg_step_number, fedavg_loss, label="FedAvg")
    fedbn = ax.plot(fedbn_step_number, fedbn_loss, label="FedBN")
    ax.legend()
    plt.axis([-3, 100, -0.1, 2.5])
    plt.ylabel("Training loss")
    plt.xlabel("Number of FL round")
    plt.title("SVHN")
    plt.savefig("convergence_rate.png")
