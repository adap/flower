"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import matplotlib.pyplot as plt
import numpy as np

from fedavgm.dataset import cifar10, partition


def plot_concentrations_cifar10():
    """Create a plot with different concentrations for dataset using LDA."""
    x_train, y_train, x_test, y_test, input_shape, num_classes = cifar10(
        10, (32, 32, 3)
    )
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    num_clients = 30

    # Simulated different concentrations for partitioning
    concentration_values = [np.inf, 100, 1, 0.1, 0.01, 1e-12]
    color = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, num_classes))
    num_plots = len(concentration_values)
    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5), sharey=True)

    pos = axs[0].get_position()
    pos.x0 += 0.1
    axs[0].set_position(pos)

    for i, concentration in enumerate(concentration_values):
        partitions = partition(x, y, num_clients, concentration, num_classes)

        for c in range(num_clients):
            _, y_client = partitions[c]
            lefts = [0]
            ax = axs[i]
            class_counts = np.bincount(y_client, minlength=num_classes)
            np.sum(class_counts > 0)

            class_distribution = class_counts.astype(np.float16) / len(y_client)

            for idx, v in enumerate(class_distribution[:-1]):
                lefts.append(lefts[idx] + v)

            ax.barh(c, class_distribution, left=lefts, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Class distribution")
            ax.set_title(f"Concentration = {concentration}")

    fig.text(0, 0.5, "Client", va="center", rotation="vertical")
    plt.tight_layout()
    plt.savefig("../docs/concentration_cifar10_v2.png")
    print(">>> Concentration plot created")
    return None


if __name__ == "__main__":
    plot_concentrations_cifar10()
