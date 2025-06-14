#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""# ------------------------------------------------------------------------ Created on
Thu Oct 24 14:17:57 2024

@author: Zeyneddin Oz

# Coded for plotting Federated Learning results which uses WEAR dataset.
# E-Mail: zeyneddin.oez@uni-siegen.de
# ------------------------------------------------------------------------
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .utils_for_tinyhar import get_learning_type_name


def plot_all_sbj_using_line_chart(
    df: pd.DataFrame, directory_name, save_fig=True
) -> plt:

    # Plot all columns in the DataFrame
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    for column in df.columns:
        plt.plot(df[column], label=f"{column}")

    title = "F1-score Performance of all clients during FL training"

    plt.title(title, fontweight="bold")
    plt.xlabel("Global round", fontweight="bold")
    plt.ylabel("F1-score", fontweight="bold")
    plt.legend(loc="upper right")  # You can adjust the location of the legend
    plt.grid(True)

    if save_fig:
        file_path = os.path.join(directory_name, title + ".png")
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()


def plot_one_sbj_using_line_chart(
    df: pd.DataFrame(), client_num: int, directory_name: os.path, save_fig=False
) -> plt:

    # Plot a specific column, for example, the first column (index 0)
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    plt.plot(
        df.iloc[:, client_num], label="Client_" + str(client_num)
    )  # Change the index or name to select a different column

    title = "Line Chart of Client_" + str(client_num) + " during FL training"

    plt.title(title, fontweight="bold")
    plt.xlabel("Global round", fontweight="bold")
    plt.ylabel("F1-score", fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True)

    if save_fig:
        file_path = os.path.join(directory_name, title + ".png")
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()


def plot_LMs_means_per_global_round(
    df: pd.DataFrame, GLOBAL_ROUND: int, directory_name, save_fig=True
) -> plt:

    LMs_means_per_global_round = []

    for g_round in range(GLOBAL_ROUND):  # for g_round in range(GLOBAL_ROUND+1):
        LMs_means_per_global_round.append(np.mean(df.loc[g_round].values))

    # Plot a specific column, for example, the first column (index 0)
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(
        LMs_means_per_global_round, label="LMs means"
    )  # Change the index or name to select a different column

    title = f"F1-score mean accross all LMs and labels during {GLOBAL_ROUND} global round in FL"

    plt.title(title, fontweight="bold")
    plt.xlabel("Global round", fontweight="bold")
    plt.ylabel("F1-score", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_fig:
        file_path = os.path.join(directory_name, title + ".png")
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()


def plot_GM_performance_after_FL_training(
    df: pd.DataFrame, GLOBAL_ROUND: int, directory_name, save_fig=True
) -> plt:
    """Plot GM performance after a specified global round as a bar chart.

    Args:
        df (pd.DataFrame): DataFrame containing F1 scores for each client.
        GLOBAL_ROUND (int): The global round index to plot.
        directory_name (str): Directory to save the plot if save_fig=True.
        save_fig (bool): Whether to save the figure to a file. Defaults to True.
    """

    categories = df.columns.values
    values = df.iloc[GLOBAL_ROUND - 1].values

    # Calculate the mean value
    mean_value = np.mean(values)

    # Create a figure with a larger size
    plt.figure(figsize=(18, 6))  # Width: 18, Height: 6

    # Create a bar chart
    bars = plt.bar(categories, values, color="skyblue", edgecolor="black", width=0.6)

    title = f"GM performance after {GLOBAL_ROUND} global round in FL"

    # Add labels and title
    plt.xlabel("Clients (Sbjs)", fontsize=14, fontweight="bold")
    plt.ylabel("F1-scores", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Set y-axis ticks to display all values between 0.0 and 1.0 in steps of 0.1
    y_ticks = np.arange(0, 1.1, 0.1)  # Ensure it includes 1.0
    plt.yticks(y_ticks, fontsize=12)

    # Display values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            round(yval, 2),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add horizontal line for the mean
    plt.axhline(
        y=mean_value,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_value:.2f}",
    )

    # Add legend for the mean line
    plt.legend(fontsize=12)

    if save_fig:
        file_path = os.path.join(directory_name, title + ".png")
        plt.savefig(file_path, bbox_inches="tight")

    # Show the chart
    plt.show()


def plot_multi_bar_chart(
    F1Score_dict_all: dict, LEARNING_TYPES_LIST: list, directory_name, save_fig=True
) -> plt:

    # Fix learning type name style
    # LEARNING_TYPES_LIST = [get_learning_type_name(LEARNING_TYPE) for LEARNING_TYPE in LEARNING_TYPES_LIST]
    LEARNING_TYPES_LIST = ["LL", "CL", "FL"]

    # Extract clients and their values
    clients = list(F1Score_dict_all.keys())
    values = np.array(list(F1Score_dict_all.values()))

    # Define x positions for grouped bars
    x = np.arange(len(clients))
    bar_width = 0.25

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(16, 9))
    bar_handles = []  # Store bar handles for the legend
    for i in range(values.shape[1]):  # Iterate over the 3 values per client
        bars = ax.bar(
            x + i * bar_width,
            values[:, i],
            width=bar_width,
            label=LEARNING_TYPES_LIST[i],
        )
        bar_handles.append(bars[0])  # Add one bar from each group for the legend

    plt.yticks(np.arange(0, 1.1, 0.1))

    title = "F1Scores of Subjects Over Different Learning Systems"

    # Add labels, legend, and grid
    ax.set_xlabel("Subjects", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1Scores", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(x + bar_width)  # Adjust x-ticks to align with groups
    ax.set_xticklabels(clients, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Calculate mean values and draw horizontal lines
    LL_mean_value = np.mean([values[0] for values in F1Score_dict_all.values()])
    CL_mean_value = np.mean([values[1] for values in F1Score_dict_all.values()])
    FL_mean_value = np.mean([values[2] for values in F1Score_dict_all.values()])

    # Draw horizontal lines
    ax.axhline(y=LL_mean_value, color="blue", linestyle="--", linewidth=1.5)
    ax.axhline(y=CL_mean_value, color="orange", linestyle="--", linewidth=1.5)
    ax.axhline(y=FL_mean_value, color="green", linestyle="--", linewidth=1.5)

    # Create custom legend entries for horizontal lines
    line_handles = [
        Line2D(
            [0],
            [0],
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"LL mean: {LL_mean_value:.2f}",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"CL mean: {CL_mean_value:.2f}",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"FL mean: {FL_mean_value:.2f}",
        ),
    ]

    # Combine bar and line handles for a unified legend
    combined_handles = bar_handles + line_handles
    combined_labels = LEARNING_TYPES_LIST + [
        f"LL mean: {LL_mean_value:.2f}",
        f"CL mean: {CL_mean_value:.2f}",
        f"FL mean: {FL_mean_value:.2f}",
    ]

    ax.legend(combined_handles, combined_labels, loc="upper left")

    plt.tight_layout()

    if save_fig:
        file_path = os.path.join(directory_name, title + ".png")
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()


def plot_label_amount_as_window(
    training_dataloader_all,
    testing_dataloader_all,
    client_num: int,
    labels_set: dict,
    directory_name: os.path,
    save_fig=False,
) -> plt:

    all_labels_train = []
    all_labels_test = []

    for batch_idx, (features, labels) in enumerate(training_dataloader_all[client_num]):
        all_labels_train.extend(labels.cpu().numpy())

    for batch_idx, (features, labels) in enumerate(testing_dataloader_all[client_num]):
        all_labels_test.extend(labels.cpu().numpy())

    reversed_labels_set = {v: k for k, v in labels_set.items()}

    test_labels = np.array([reversed_labels_set[l] for l in all_labels_test])
    train_labels = np.array([reversed_labels_set[l] for l in all_labels_train])

    unique_labels = labels_set.keys()

    train_counts = [np.sum(train_labels == label) for label in unique_labels]
    test_counts = [np.sum(test_labels == label) for label in unique_labels]

    # Set up the bar chart
    bar_width = 0.35  # Width of the bars
    x = np.arange(len(unique_labels))  # Label locations

    # Create a figure with a larger size
    plt.figure(figsize=(28, 6))

    # Create bars for training and testing data
    bars1 = plt.bar(
        x - bar_width / 2,
        train_counts,
        bar_width,
        label="Training",
        color="skyblue",
        edgecolor="black",
    )
    bars2 = plt.bar(
        x + bar_width / 2,
        test_counts,
        bar_width,
        label="Testing",
        color="lightgreen",
        edgecolor="black",
    )

    title = (
        f"Label Distribution Comparison: Training vs. Testing for Client_{client_num}"
    )

    # Add labels and title with bold font
    plt.xlabel("Labels", fontsize=14, fontweight="bold")
    plt.ylabel("Count", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(x, unique_labels, rotation=-45)  # Rotate x-tick labels vertically
    plt.legend()  # Show the legend

    # Display values on top of the bars
    for bar in bars1 + bars2:  # Iterate through both sets of bars
        yval = bar.get_height()  # Get the height of the bar
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            yval,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Show the chart
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()  # Adjust layout to make room for labels

    if save_fig:
        file_path = os.path.join(directory_name, title + ".png")
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()


def plot_heat_map_of_table(
    label_based_result_table: pd.DataFrame,
    directory_name: os.path,
    save_fig=False,
    learning_type_name=None,
) -> plt:

    # Fix learning type name style
    learning_type_name = get_learning_type_name(learning_type_name)

    # Set the figure size
    plt.figure(figsize=(12, 8))

    # Create the heatmap
    sns.heatmap(
        label_based_result_table,
        annot=True,  # Show the F1-scores in each cell
        fmt=".2f",  # Format the numbers with two decimal places
        cmap="coolwarm",  # Use a diverging colormap to highlight low and high values
        cbar_kws={"label": "F1 Score"},
    )  # Add a color bar label

    title = f"{learning_type_name} - F1-Score Heatmap"

    # Add titles and labels
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Labels", fontsize=12, fontweight="bold")
    plt.ylabel("Clients", fontsize=12, fontweight="bold")

    # Show the plot
    plt.tight_layout()

    if save_fig:
        file_path = os.path.join(directory_name, title + ".png")
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()


class advanced_plot:

    # Column names:
    columns = [
        "RA_x",
        "RA_y",
        "RA_z",
        "RL_x",
        "RL_y",
        "RL_z",
        "LL_x",
        "LL_y",
        "LL_z",
        "LA_x",
        "LA_y",
        "LA_z",
    ]

    def sensor_based_analysis(
        user_test_features_all: [np.array],
        real_labels_all: list,
        pred_labels_all: list,
        client_num: int,
        window_size: int,
        directory_name,
        labels_set,
        separate_results=False,
        save_fig=True,
        learning_type_name=None,
    ) -> plt:
        """4 sonsors are subplotted and Grounth Truth vs Predicted Labels can be
        analysed via colour map."""

        data = user_test_features_all[client_num]

        # Number to labels:
        reversed_labels_set = {v: k for k, v in labels_set.items()}

        real_labels = [reversed_labels_set[l] for l in real_labels_all[client_num]]
        pred_labels = [reversed_labels_set[l] for l in pred_labels_all[client_num]]

        num_windows = data.shape[0] // window_size
        all_labels = list(labels_set.keys())
        color_map = plt.cm.get_cmap("tab20", len(all_labels))

        # Fixed color map for all labels
        label_colors = {label: color_map(i) for i, label in enumerate(all_labels)}

        def plot_sensor_data(ax, sensor_data, sensor_name):
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                ax.plot(
                    range(start, end),
                    sensor_data[start:end, 0],
                    label=f"{sensor_name}_x" if i == 0 else "",
                    color="blue",
                )
                ax.plot(
                    range(start, end),
                    sensor_data[start:end, 1],
                    label=f"{sensor_name}_y" if i == 0 else "",
                    color="orange",
                )
                ax.plot(
                    range(start, end),
                    sensor_data[start:end, 2],
                    label=f"{sensor_name}_z" if i == 0 else "",
                    color="green",
                )
                if i < num_windows - 1:
                    ax.axvline(end, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlim(0, data.shape[0])

        fig, axs = plt.subplots(
            5, 1, figsize=(35, 12), gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5]}
        )
        fig.subplots_adjust(hspace=0.4, left=0.05, right=0.98)

        plot_sensor_data(axs[0], data[:, :3], "RA")
        axs[0].set_title("Right Arm Sensor", fontweight="bold")
        axs[0].legend(loc="upper right")

        plot_sensor_data(axs[1], data[:, 3:6], "RL")
        axs[1].set_title("Right Leg Sensor", fontweight="bold")
        axs[1].legend(loc="upper right")

        plot_sensor_data(axs[2], data[:, 6:9], "LL")
        axs[2].set_title("Left Leg Sensor", fontweight="bold")
        axs[2].legend(loc="upper right")

        plot_sensor_data(axs[3], data[:, 9:12], "LA")
        axs[3].set_title("Left Arm Sensor", fontweight="bold")
        axs[3].legend(loc="upper right")

        real_label_array = np.array(
            [all_labels.index(label) for label in real_labels[:num_windows]]
        )
        pred_label_array = np.array(
            [all_labels.index(label) for label in pred_labels[:num_windows]]
        )

        axs[4].imshow(
            [real_label_array, pred_label_array],
            aspect="auto",
            cmap=ListedColormap([label_colors[label] for label in all_labels]),
            extent=[0, num_windows * window_size, -1, 1],
        )
        axs[4].set_yticks([0, 1])
        axs[4].set_yticklabels(["Predicted Labels", "Ground truth"], fontweight="bold")
        axs[4].set_xticks(range(0, num_windows * window_size, window_size))
        axs[4].set_xticklabels([])

        for i in range(1, num_windows):
            axs[4].axvline(
                i * window_size, color="black", linestyle="--", linewidth=0.5
            )

        if separate_results:
            axs[4].axhline(0, color="black", linewidth=0.5)

        legend_handles = [
            Patch(color=label_colors[label], label=label) for label in all_labels
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(all_labels))

        fig.suptitle(
            f"{learning_type_name} - Ground Truth vs Predictions of Client_{client_num}",
            fontsize=36,
            fontweight="bold",
        )

        if save_fig:
            file_path = os.path.join(
                directory_name, f"Ground Truth vs Predictions of Client_{client_num}"
            )
            plt.savefig(file_path)

        plt.show()

    def sensor_based_analysis_with_all_learning_systems(
        user_test_features_all: [np.array],
        real_labels_all: list,
        pred_labels_all_LL: list,
        pred_labels_all_CL: list,
        pred_labels_all_FL: list,
        client_num: int,
        window_size: int,
        labels_set,
        directory_name,
        save_fig=True,
    ) -> plt:
        """4 sensors are subplotted, and Ground Truth vs Predicted Labels can be
        analyzed via color map."""

        data = user_test_features_all[client_num]

        # Number to labels:
        reversed_labels_set = {v: k for k, v in labels_set.items()}

        real_labels = [reversed_labels_set[l] for l in real_labels_all[client_num]]
        pred_labels_LL = [
            reversed_labels_set[l] for l in pred_labels_all_LL[client_num]
        ]
        pred_labels_CL = [
            reversed_labels_set[l] for l in pred_labels_all_CL[client_num]
        ]
        pred_labels_FL = [
            reversed_labels_set[l] for l in pred_labels_all_FL[client_num]
        ]

        num_windows = data.shape[0] // window_size
        all_labels = list(labels_set.keys())
        color_map = plt.cm.get_cmap("tab20", len(all_labels))

        # Fixed color map for all labels
        label_colors = {label: color_map(i) for i, label in enumerate(all_labels)}

        def plot_sensor_data(ax, sensor_data, sensor_name):
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                ax.plot(
                    range(start, end),
                    sensor_data[start:end, 0],
                    label=f"{sensor_name}_x" if i == 0 else "",
                    color="blue",
                )
                ax.plot(
                    range(start, end),
                    sensor_data[start:end, 1],
                    label=f"{sensor_name}_y" if i == 0 else "",
                    color="orange",
                )
                ax.plot(
                    range(start, end),
                    sensor_data[start:end, 2],
                    label=f"{sensor_name}_z" if i == 0 else "",
                    color="green",
                )
                if i < num_windows - 1:
                    ax.axvline(end, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlim(0, data.shape[0])

        fig, axs = plt.subplots(
            5, 1, figsize=(35, 12), gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5]}
        )
        fig.subplots_adjust(hspace=0.4, left=0.05, right=0.98)

        plot_sensor_data(axs[0], data[:, :3], "RA")
        axs[0].set_title("Right Arm Sensor", fontweight="bold")
        axs[0].legend(loc="upper right")

        plot_sensor_data(axs[1], data[:, 3:6], "RL")
        axs[1].set_title("Right Leg Sensor", fontweight="bold")
        axs[1].legend(loc="upper right")

        plot_sensor_data(axs[2], data[:, 6:9], "LL")
        axs[2].set_title("Left Leg Sensor", fontweight="bold")
        axs[2].legend(loc="upper right")

        plot_sensor_data(axs[3], data[:, 9:12], "LA")
        axs[3].set_title("Left Arm Sensor", fontweight="bold")
        axs[3].legend(loc="upper right")

        # Prepare arrays for labels
        real_label_array = np.array(
            [all_labels.index(label) for label in real_labels[:num_windows]]
        )
        pred_label_array_LL = np.array(
            [all_labels.index(label) for label in pred_labels_LL[:num_windows]]
        )
        pred_label_array_CL = np.array(
            [all_labels.index(label) for label in pred_labels_CL[:num_windows]]
        )
        pred_label_array_FL = np.array(
            [all_labels.index(label) for label in pred_labels_FL[:num_windows]]
        )

        # Combine arrays for the imshow plot
        combined_labels = np.vstack(
            [
                real_label_array,
                pred_label_array_LL,
                pred_label_array_CL,
                pred_label_array_FL,
            ]
        )

        axs[4].imshow(
            combined_labels,
            aspect="auto",
            cmap=ListedColormap([label_colors[label] for label in all_labels]),
            extent=[0, num_windows * window_size, -4, 0],
        )
        axs[4].set_yticks([-3.5, -2.5, -1.5, -0.5])
        axs[4].set_yticklabels(
            [
                "Federated Learning",
                "Centralized Learning",
                "Local Learning",
                "Ground Truth",
            ],
            fontweight="bold",
        )
        axs[4].set_xticks(range(0, num_windows * window_size, window_size))
        axs[4].set_xticklabels([])

        for i in range(1, num_windows):
            axs[4].axvline(
                i * window_size, color="black", linestyle="--", linewidth=0.5
            )

        # NEW ADDED. ################################################
        # Add horizontal lines to divide the label arrays
        for i in range(1, combined_labels.shape[0]):  # combined_labels.shape[0] is 4
            axs[4].axhline(-i, color="black", linewidth=0.5)
        # NEW ADDED. ################################################

        legend_handles = [
            Patch(color=label_colors[label], label=label) for label in all_labels
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(all_labels))

        fig.suptitle(
            f"Ground Truth vs Predictions of Client_{client_num} Over All Learning Systems",
            fontsize=36,
            fontweight="bold",
        )

        if save_fig:
            file_path = os.path.join(
                directory_name, f"Ground Truth vs Predictions of Client_{client_num}"
            )
            plt.savefig(file_path)

        plt.show()

    def interactive_analysis(
        user_test_features_all: [np.array],
        real_labels_all: list,
        pred_labels_all: list,
        client_num: int,
        window_size: int,
        directory_name,
        labels_set,
        save_fig=False,
    ) -> go.Figure:
        """An interactive analysis of all sensors together in a line graph with
        advantages of zooming a specific range."""
        data = user_test_features_all[client_num]

        # Number to labels:
        reversed_labels_set = {v: k for k, v in labels_set.items()}

        real_labels = [reversed_labels_set[l] for l in real_labels_all[client_num]]
        pred_labels = [reversed_labels_set[l] for l in pred_labels_all[client_num]]

        num_windows = data.shape[0] // window_size
        fig = go.Figure()

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_data = data[start:end]

            for j, column_name in enumerate(advanced_plot.columns):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(start, end)),
                        y=window_data[:, j],
                        mode="lines",
                        name=column_name if i == 0 else "",
                        line=dict(width=1),
                    )
                )

            if i < num_windows - 1:
                fig.add_vline(x=end, line=dict(color="black", width=1, dash="dash"))

            midpoint = start + window_size // 2
            fig.add_annotation(
                x=midpoint,
                y=9,
                text=f"W{i}",
                showarrow=False,
                font=dict(color="black", size=10),
            )
            fig.add_annotation(
                x=midpoint,
                y=-5,
                text=f"Real: {real_labels[i]}",
                showarrow=False,
                font=dict(color="blue", size=10),
            )
            fig.add_annotation(
                x=midpoint,
                y=-7,
                text=f"Pred: {pred_labels[i]}",
                showarrow=False,
                font=dict(color="red", size=10),
            )

        title = "Signals Plotted in Windows with Real and Predicted Labels"

        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=True),
            ),
            yaxis_title="Signal Value",
            width=1400,
            height=600,
            showlegend=True,
        )

        if save_fig:
            file_path = os.path.join(directory_name, f"{title}.png")
            fig.write_image(file_path)
            print(f"Interactive plot saved to {file_path}")

        fig.show()


class common_plots:
    # Create default categories if none are provided
    categories = ["sbj_" + str(i) for i in range(24)]

    def plot_clients_performance_after_training(
        final_fscores,
        training_round,
        directory_name=None,
        save_fig=False,
        learning_type_name=None,
    ):
        """Plot a bar chart with values, add numbers on the bars, and draw a mean line.

        Args:
            final_fscores (list or np.ndarray): List of F1 scores to plot.
            training_round (int): Current training round for the title.
            directory_name (str, optional): Directory to save the figure if save_fig=True.
            save_fig (bool, optional): Whether to save the figure. Defaults to False.
        """

        # Calculate mean
        mean_value = np.mean(final_fscores)

        title = (
            f"{learning_type_name} - Performance after {training_round} Training Round"
        )

        # Create a figure with a larger size
        plt.figure(figsize=(18, 6))  # Width: 18, Height: 6

        # Create the bar chart
        bars = plt.bar(
            common_plots.categories,
            final_fscores,
            color="skyblue",
            edgecolor="black",
            width=0.6,
        )

        # Add labels, title, and grid
        plt.xlabel("Clients (Sbjs)", fontsize=14, fontweight="bold")
        plt.ylabel("F1-Scores", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add finer y-axis ticks
        y_ticks = np.arange(0, 1.05, 0.1)  # Tick range from 0 to 1, step 0.1
        plt.yticks(y_ticks, fontsize=12)

        # Add value labels on bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Add horizontal mean line
        plt.axhline(
            y=mean_value,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_value:.2f}",
        )

        # Add legend
        plt.legend(fontsize=12)

        # Save the figure if required
        if save_fig and directory_name:
            import os

            file_path = os.path.join(directory_name, f"{title}.png")
            plt.savefig(file_path, bbox_inches="tight")

        # Show the plot
        plt.show()
