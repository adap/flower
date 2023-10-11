import argparse
import os
import pickle
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

def check_folders(directory, required_folders):
    for folder in required_folders:
        folder_path = os.path.join(directory, folder)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Error: The folder '{folder}' is missing in the input directory '{directory}'.")
            exit(1)

def get_accuracies(path):
    with open(path, 'rb') as f:
            hist = pickle.load(f)['history']

    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])
    return np.asarray(rounds), np.asarray(values) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_type", choices=["table", "figure"])
    parser.add_argument("--output_path")

    if parser.parse_known_args()[0].output_type == "table":
        parser.add_argument("--input_directory")
    elif parser.parse_known_args()[0].output_type == "figure":
        parser.add_argument("--dataset_name")
        parser.add_argument("--input_directory")

    args = parser.parse_args()

    if args.output_type == "table":
        input_directory = args.input_directory
        
        required_folders = [
            "femnist_0", "femnist_1", "femnist_2",
            "cifar10_0", "cifar10_1", "cifar10_2",
            "cifar100_0", "cifar100_1", "cifar100_2"
        ]

        check_folders(input_directory, required_folders)

        femnist = [f'{max(accs):.2f}' for rnds, accs in [get_accuracies(f'{input_directory}/femnist_{i}/results.pkl') for i in [0, 1, 2]]]
        cifar10 = [f'{max(accs):.2f}' for rnds, accs in [get_accuracies(f'{input_directory}/cifar10_{i}/results.pkl') for i in [0, 1, 2]]]
        cifar100 = [f'{max(accs):.2f}' for rnds, accs in [get_accuracies(f'{input_directory}/cifar100_{i}/results.pkl') for i in [0, 1, 2]]]

        fig, ax = plt.subplots()
        table = ax.table(
            cellText=[list(i) for i in zip(*[femnist, cifar10, cifar100])],
            cellLoc="center",
            colLabels=['FEMNIST', 'CIFAR10', 'CIFAR100'],
            rowLabels=["FedAvg", "NaiveMix", "FedMix"],
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        ax.axis("off")
        plt.savefig(args.output_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    elif args.output_type == "figure":
        dataset_name = args.dataset_name
        input_directory = args.input_directory
    
        required_folders = [f"{dataset_name}_0", f"{dataset_name}_1", f"{dataset_name}_2"]
    
        check_folders(input_directory, required_folders)

        all_rounds, all_accuracies, max_accuracies = {}, {}, {}
        for i, lbl in enumerate(["FedAvg", "NaiveMix", "FedMix"]):
            all_rounds[lbl], all_accuracies[lbl] = get_accuracies(f'{input_directory}/{dataset_name}_{i}/results.pkl')
            max_accuracies[lbl] = max(all_accuracies[lbl])

        def average_list_values(values, n=10):
            values = [values[0], ] * n + list(values) + [values[-1], ] * n
            averaged_values = []
            for i, v in enumerate(values[n:-n]):
                averaged_values.append(mean((values[i:i+2*n])))

            return averaged_values

        for (lbl, accuracies), (_, rounds) in zip(all_accuracies.items(), all_rounds.items()):
            plt.plot(rounds, average_list_values(accuracies, 5), label=lbl)

        plt.xlabel('Comm. Round')
        plt.ylabel('Test Acc. (%)')
        plt.text(0.7*len(rounds), 50*max(accuracies), 'Max Accuracy:\n' + '\n'.join([f'{l}: {a:.2f}' for l, a in max_accuracies.items()]))
        plt.legend()

        plt.savefig(args.output_path)
        plt.close()

if __name__ == "__main__":
    main()
