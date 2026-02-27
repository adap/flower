import matplotlib.pyplot as plt
import wandb
import os
import numpy as np


def cosine_plots(num_clients, num_freeriders, metrics_distributed_fit, dataset):
    distribution = False
    member_data = {client_id: [] for client_id in range(num_clients)}
    nonmember_data = {client_id: [] for client_id in range(num_clients)}
    epochs = len(metrics_distributed_fit["cosine_metrics"])
    print(metrics_distributed_fit["cosine_metrics"])
    for round, (dist_member, dist_nonmember) in metrics_distributed_fit["cosine_metrics"]:
        for client_id in range(num_clients):
            member_epoch_data = np.array(dist_member[client_id])
            nonmember_epoch_data = np.array(dist_nonmember[client_id])
            member_data[client_id].append(member_epoch_data)
            nonmember_data[client_id].append(nonmember_epoch_data)

        if round % 15 == 0 and distribution:
            fig, axes = plt.subplots(nrows=1, ncols=num_clients, figsize=(5 * ((num_clients + 1)), 7))
            fig.suptitle(f"Round {round} - Cosine distribution of Member and Non-member subsets")

            for client_id in range(num_clients):
                col = client_id
                ax = axes[col] if num_clients > 1 else axes

                ax.hist(dist_member[client_id], bins=25, alpha=0.5, label="Member", color="blue")
                ax.hist(dist_nonmember[client_id], bins=25, alpha=0.5, label="Non-member", color="red")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                if client_id < num_freeriders:
                    ax.set_title("Free-rider")
                else:
                    ax.set_title("Benign client")
                ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(wandb.run.dir, "distribution_sim.pdf"))

    fig, axs = plt.subplots(num_clients, 3, figsize=(15, 5 * num_clients))

    global_min_dist = float("inf")
    global_max_dist = float("-inf")
    global_min_avg = float("inf")
    global_max_avg = float("-inf")
    global_min_std = float("inf")
    global_max_std = float("-inf")

    for client_id in range(num_clients):
        member_data_client = np.array(member_data[client_id])
        nonmember_data_client = np.array(nonmember_data[client_id])

        member_means = np.mean(member_data_client, axis=1)
        nonmember_means = np.mean(nonmember_data_client, axis=1)
        member_stds = np.std(member_data_client, axis=1)
        nonmember_stds = np.std(nonmember_data_client, axis=1)

        dist_min = min(np.min(member_means - member_stds), np.min(nonmember_means - nonmember_stds))
        dist_max = max(np.max(member_means + member_stds), np.max(nonmember_means + nonmember_stds))
        global_min_dist = min(global_min_dist, dist_min)
        global_max_dist = max(global_max_dist, dist_max)

        avg_min = min(np.min(member_means), np.min(nonmember_means))
        avg_max = max(np.max(member_means), np.max(nonmember_means))
        global_min_avg = min(global_min_avg, avg_min)
        global_max_avg = max(global_max_avg, avg_max)

        std_min = min(np.min(member_stds), np.min(nonmember_stds))
        std_max = max(np.max(member_stds), np.max(nonmember_stds))
        global_min_std = min(global_min_std, std_min)
        global_max_std = max(global_max_std, std_max)

    for client_id in range(num_clients):
        if client_id < num_freeriders:
            name = "Freerider"
        else:
            name = "Benign"
        member_data_client = np.array(member_data[client_id])
        nonmember_data_client = np.array(nonmember_data[client_id])

        member_means = np.mean(member_data_client, axis=1)
        nonmember_means = np.mean(nonmember_data_client, axis=1)
        member_stds = np.std(member_data_client, axis=1)
        nonmember_stds = np.std(nonmember_data_client, axis=1)

        # Distribution plot
        for epoch in range(epochs):
            axs[client_id, 0].errorbar(epoch, member_means[epoch], yerr=member_stds[epoch], fmt="-", color="blue", alpha=0.5)
            axs[client_id, 0].errorbar(
                epoch, nonmember_means[epoch], yerr=nonmember_stds[epoch], fmt="-", color="red", alpha=0.5
            )

        axs[client_id, 0].set_title("Cosine similarity distribution - {} - {}".format(dataset, name))
        axs[client_id, 0].set_xlabel("epochs")
        axs[client_id, 0].set_ylabel("cosine similarity")
        axs[client_id, 0].legend(["members", "nonmembers"])
        axs[client_id, 0].set_ylim(global_min_dist, global_max_dist)

        axs[client_id, 1].plot(list(range(epochs)), member_means, label="member", color="blue")
        axs[client_id, 1].plot(list(range(epochs)), nonmember_means, label="nonmember", color="red")
        axs[client_id, 1].set_title("Cosine similarity avg - {} - {}".format(dataset, name))
        axs[client_id, 1].set_xlabel("epochs")
        axs[client_id, 1].legend()
        axs[client_id, 1].set_ylim(global_min_avg, global_max_avg)

        axs[client_id, 2].plot(list(range(epochs)), member_stds, label="member", color="blue")
        axs[client_id, 2].plot(list(range(epochs)), nonmember_stds, label="nonmember", color="red")
        axs[client_id, 2].set_title("Cosine similarity std - {} - {}".format(dataset, name))
        axs[client_id, 2].set_xlabel("epochs")
        axs[client_id, 2].legend()
        axs[client_id, 2].set_ylim(global_min_std, global_max_std)

    plt.tight_layout()
    plt.savefig(os.path.join(wandb.run.dir, "cosine_sim_all_clients.pdf"))


def yeom_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit):
    plt.figure(figsize=(15, 6))
    detection_loss_clients = {client: [] for client in range(num_clients)}
    for round, losses in metrics_distributed_fit["detection_loss"]:
        for i, loss in enumerate(losses):
            detection_loss_clients[i].append(loss)

    for client_id in range(num_clients):
        if client_id < num_freeriders:
            fr = "Free-Rider"
        else:
            fr = "Honest client"
        plt.plot(range(num_rounds), detection_loss_clients[client_id], label=f"{fr}")

    plt.title("Clients canary losses")
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(wandb.run.dir,'canary_loss.pdf'))


def l2_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit, dataset):
    plt.figure(figsize=(15, 6))
    detection_l2_clients = {client: [] for client in range(num_clients)}

    for round, (l2s, zscore) in metrics_distributed_fit["l2_metrics"]:
        for i, l2 in enumerate(l2s):
            detection_l2_clients[i].append(l2)

    for client_id in range(num_clients):
        if client_id < num_freeriders:
            fr = "Free-Rider"
        else:
            fr = "Honest client"
        plt.plot(range(num_rounds), detection_l2_clients[client_id], label=f"{fr}")

    plt.title("L2 values")
    plt.grid()
    plt.ylabel("L2")
    plt.xlabel("Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(wandb.run.dir,'l2_norm.pdf'))


def std_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit, dataset):
    plt.figure(figsize=(15, 6))
    detection_l2_clients = {client: [] for client in range(num_clients)}

    for round, (l2s, zscore) in metrics_distributed_fit["std_metrics"]:
        for i, l2 in enumerate(l2s):
            detection_l2_clients[i].append(l2)

    for client_id in range(num_clients):
        if client_id < num_freeriders:
            fr = "Free-Rider"
        else:
            fr = "Honest client"
        plt.plot(range(num_rounds), detection_l2_clients[client_id], label=f"{fr}")

    plt.title("STD values")
    plt.grid()
    plt.ylabel("STD")
    plt.xlabel("Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(wandb.run.dir,'std.pdf'))


def cosim_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit, dataset):
    plt.figure(figsize=(15, 6))
    detection_cosim_clients = {client: [] for client in range(num_clients)}

    for round, (cosims, zscore) in metrics_distributed_fit["cosim_metrics"]:
        for i, cosim in enumerate(cosims):
            detection_cosim_clients[i].append(cosim)
    
    for client_id in range(num_clients):
        if client_id < num_freeriders:
            fr = "Free-Rider"
        else:
            fr = "Honest client"
        
        cosim_values = np.array(detection_cosim_clients[client_id][1:])
        rounds = np.arange(num_rounds - 1)
        valid_mask = ~np.isnan(cosim_values)
        plt.plot(rounds[valid_mask], cosim_values[valid_mask], label=f"{fr}")

    plt.title("COSIM values")
    plt.grid()
    plt.ylabel("COSIM")
    plt.xlabel("Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(wandb.run.dir, 'cosim.pdf'))


def plot_metrics(history, num_clients, num_rounds, attack_types, num_freeriders, dataset="cifar10"):
    num_metrics = len(history.metrics_distributed_fit)
    plt.figure(figsize=(15, 6))

    metrics_centralized_accuracy = history.metrics_centralized["accuracy"]
    round_weighted_accuracy = [data[0] for data in metrics_centralized_accuracy]
    acc_weighted_accuracy = [100.0 * data[1] for data in metrics_centralized_accuracy]

    plt.subplot(2, num_metrics // 2 + 1, 1)
    plt.plot(round_weighted_accuracy, acc_weighted_accuracy, label="Accuracy")
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("Global Accuracy")

    metrics_centralized_loss = history.metrics_centralized["loss"]
    round_loss = [data[0] for data in metrics_centralized_loss]
    loss_values = [data[1] for data in metrics_centralized_loss]

    plt.subplot(2, num_metrics // 2 + 1, 2)
    plt.plot(round_loss, loss_values, label="Loss", color="red")
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Round")
    plt.title("Global Loss")

    metrics_distributed_fit = history.metrics_distributed_fit
    for m, (metric_name, metric_data) in enumerate(metrics_distributed_fit.items(), start=1):
        if metric_name not in ("detection_loss", "cosine_metrics", "pia", "orthogonality_metrics", "historical_detection_results", "l2_metrics", "std_metrics", "cosim_metrics"):
            print(metric_name)
            round_metric = [data[0] for data in metric_data]
            metric_values = [data[1] for data in metric_data]
            plt.subplot(2, num_metrics // 2 + 1, m + 2)
            plt.plot(round_metric, metric_values, label=metric_name)
            plt.grid()
            plt.ylabel(metric_name)
            plt.xlabel("Round")
            plt.title(metric_name)
    plt.tight_layout()
    plt.savefig(os.path.join(wandb.run.dir,'attack_metrics.pdf'))

    if "yeom" in attack_types:
        yeom_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit)

    if "cosine" in attack_types:
        cosine_plots(num_clients, num_freeriders, metrics_distributed_fit, dataset)

    if "l2" in attack_types:
        l2_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit, dataset)

    if "std" in attack_types:
        std_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit, dataset)
    
    if "cosine_similarity" in attack_types:
        cosim_plots(num_clients, num_freeriders, num_rounds, metrics_distributed_fit, dataset)

