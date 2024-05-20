# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Label distribution plotting."""
from typing import Tuple, Optional, Union, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from metrics import compute_counts
from partitioner import Partitioner

# Constants for plot types and size units
PLOT_TYPES = ("bar", "heatmap")
SIZE_UNITS = ("absolute", "percent")
AXIS_TYPES = ("x", "y")


def plot_label_distributions(
        partitioner: Partitioner,
        label_name: str,
        plot_type: str,
        size_unit: str,
        max_num_partitions: Optional[int] = None,
        partition_id_axis: str = "x",
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = "Per Partition Label Distribution",
        cmap: Optional[Union[str, mcolors.Colormap]] = None,
        legend: bool = True,
        legend_title: str = "Labels",
        verbose_labels: bool = True,
        **plot_kwargs,
):
    """
    Plot the label distribution of the partitions.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    label_name : str
        Column name identifying label based on which the plot will be created.
    plot_type : str
        Type of plot, either "bar" or "heatmap".
    size_unit : str
        "absolute" or "percent". "absolute" - (number of samples). "percent" -
        normalizes each value, so they sum up to 100%.
    max_num_partitions : Optional[int]
        The number of partitions that will be used. If left None, then all partitions
        will be used.
    partition_id_axis : str
        "x" or "y". The axis on which the partition_id will be marked.
    ax : Optional[Axes]
        Matplotlib Axes object to plot on.
    figsize : Optional[Tuple[float, float]]
        Size of the figure.
    title : str
        Title of the plot.
    colormap : Optional[Union[str, mcolors.Colormap]]
        Colormap for the heatmap.
    legend : bool
        Include the legend.
    legend_title : str
        Name for the legend.
    verbose_labels : bool
        Whether to use verbose versions of the labels.

    Returns
    -------
    ax : Axes
        The Axes object with the plot.
    df : pd.DataFrame
        The DataFrame used for plotting.
    """

    _validate_parameters(plot_type, size_unit, partition_id_axis)

    if label_name not in partitioner.dataset.column_names:
        raise ValueError(
            f"The specified 'label_name': '{label_name}' is not present in the "
            f"dataset.")

    if max_num_partitions is None:
        max_num_partitions = partitioner.num_partitions
    else:
        max_num_partitions = min(max_num_partitions, partitioner.num_partitions)
    partitions = [partitioner.load_partition(i) for i in range(max_num_partitions)]

    partition = partitions[0]
    try:
        unique_labels = partition.features[label_name].str2int(
            partition.features[label_name].names)
    except AttributeError: # If the label_name is not formally a Label
        unique_labels = partitioner.dataset.unique(label_name)
    num_labels = len(unique_labels)

    partition_id_to_label_absolute_size = {
        pid: compute_counts(partition[label_name], unique_labels)
        for pid, partition in enumerate(partitions)
    }

    df = pd.DataFrame.from_dict(partition_id_to_label_absolute_size, orient="index")
    df.index.name = "Partition ID"

    if size_unit == "percent":
        df = df.div(df.sum(axis=1), axis=0) * 100.0

    if partition_id_axis == "x" and plot_type == "heatmap":
        df = df.T

    xlabel, ylabel = _initialize_xy_labels(plot_type, size_unit, partition_id_axis)
    cbar_title = _initialize_cbar_title(plot_type, size_unit)
    figsize = _initialize_figsize(figsize, plot_type, partition_id_axis, max_num_partitions, num_labels)

    ax = _plot_data(df, plot_type, ax, figsize, title, cmap, xlabel, ylabel,
                    cbar_title, legend, verbose_labels, legend_title, partition,
                    label_name, **plot_kwargs)

    return ax, df


def _initialize_xy_labels(plot_type: str, size_unit: str, partition_id_axis: str) -> Tuple[str, str]:
    if plot_type == "bar":
        xlabel = "Partition ID"
        ylabel = "Count" if size_unit == "absolute" else "Percent %"
    elif plot_type == "heatmap":
        xlabel = "Partition ID"
        ylabel = "Label"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'bar' or 'heatmap'.")

    if partition_id_axis == "y":
        xlabel, ylabel = ylabel, xlabel

    return xlabel, ylabel


def _validate_parameters(plot_type: str, size_unit: str, partition_id_axis: str):
    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}.")
    if size_unit not in SIZE_UNITS:
        raise ValueError(
            f"Invalid size_unit: {size_unit}. Must be one of {SIZE_UNITS}.")
    if partition_id_axis not in AXIS_TYPES:
        raise ValueError(
            f"Invalid partition_id_axis: {partition_id_axis}. Must be 'x' or 'y'.")


def _initialize_cbar_title(plot_type: str, size_unit: str) -> Optional[str]:
    if plot_type == "heatmap":
        return "Count" if size_unit == "absolute" else "Percent %"
    return None


def _initialize_figsize(figsize: Optional[Tuple[int, int]], plot_type: str,
                        partition_id_axis: str, num_partitions: int, num_labels: int) -> Tuple[
    float, float]:
    if figsize is not None:
        return figsize

    if plot_type == "bar":
        if partition_id_axis == "x":
            figsize = (6.4, 4.8)
        elif partition_id_axis == "y":
            figsize = (6.4, np.sqrt(num_partitions))
    elif plot_type == "heatmap":
        if partition_id_axis == "x":
            figsize = (3 * np.sqrt(num_partitions), np.sqrt(num_labels))
        elif partition_id_axis == "y":
            figsize = (3 * np.sqrt(num_labels), np.sqrt(num_partitions))

    return figsize


def _plot_data(df: pd.DataFrame, plot_type: str, ax: Optional[Axes],
               figsize: Tuple[float, float], title: str, colormap, xlabel: str,
               ylabel: str, cbar_title: str, legend: bool, verbose_labels: bool,
               legend_title: str, partition, label_name: str, **plot_kwargs) -> Axes:
    if plot_type == "bar":
        return _plot_bar(df, ax, figsize, title, colormap, xlabel, ylabel, legend,
                         legend_title, verbose_labels, partition, label_name,
                         **plot_kwargs)
    elif plot_type == "heatmap":
        return _plot_heatmap(df, ax, figsize, title, colormap, xlabel, ylabel,
                             cbar_title, legend, **plot_kwargs)


def _plot_bar(df: pd.DataFrame, ax: Optional[Axes], figsize: Tuple[float, float],
              title: str, colormap, xlabel: str, ylabel: str, legend: bool,
              legend_title: str, verbose_labels: bool, partition, label_name: str,
              **plot_kwargs) -> Axes:
    if colormap is None:
        colormap = "RdYlGn"
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    kind = "bar" if xlabel == "Partition ID" else "barh"
    ax = df.plot(kind=kind, stacked=True, ax=ax, title=title, legend=False,
                 colormap=colormap, rot=0,
                 **plot_kwargs)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    xticklabels = ax.get_xticklabels()
    if len(xticklabels) > 20:
        # Make every other xtick label not visible
        for i, label in enumerate(xticklabels):
            if i % 2 == 1:
                label.set_visible(False)

    if legend:
        handles, legend_labels = ax.get_legend_handles_labels()
        if verbose_labels:
            try:
                legend_names = partition.features[label_name].int2str(
                    [int(v) for v in legend_labels])
            except AttributeError:
                legend_names = legend_labels
        else:
            legend_names = legend_labels

        _ = ax.figure.legend(handles[::-1], legend_names[::-1], title=legend_title,
                             loc="outside center right", bbox_to_anchor=(1.3, 0.5))

    return ax


def _plot_heatmap(df: pd.DataFrame, ax: Optional[Axes], figsize: Tuple[float, float],
                  title: str, colormap, xlabel: str, ylabel: str, cbar_title: str,
                  legend: bool, **plot_kwargs) -> Axes:
    if colormap is None:
        colormap = sns.light_palette("seagreen", as_cmap=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    fmt = ",d" if "absolute" in df.columns else "0.2f"
    sns.heatmap(df, ax=ax, cmap=colormap, fmt=fmt, cbar=legend,
                cbar_kws={'label': cbar_title}, **plot_kwargs)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    return ax

def _initialize_comparison_figsize(figsize: Optional[Tuple[float, float]], num_partitioners: int) -> Tuple[
    float, float]:
    if figsize is not None:
        return figsize
    x = 4 + (num_partitioners - 1) * 2
    y = 4.8
    figsize = (x, y)
    return figsize

def _initialize_comparison_xy_labels(plot_type: str, partition_id_axis: str) -> Tuple[str, str]:
    if plot_type == "bar":
        xlabel = "Partition ID"
        ylabel = "Class distribution"
    elif plot_type == "heatmap":
        xlabel = "Partition ID"
        ylabel = "Label"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'bar' or 'heatmap'.")

    if partition_id_axis == "y":
        xlabel, ylabel = ylabel, xlabel

    return xlabel, ylabel


def compare_label_distribution(
        partitioner_list: List[Partitioner],
        label_name: Union[str, List[str]],
        plot_type: str = "bar",
        size_unit: str = "percent",
        max_num_partitions: Optional[Union[int]] = 30,
        partition_id_axis: str = "y",
        figsize: Optional[Tuple[float, float]] = None,
        subtitle: str = "Comparison of Per Partition Label Distribution",
        titles: Optional[List[str]] = None,
        cmap: Optional[Union[str, mcolors.Colormap]] = None,
        legend: bool = False,
        legend_title: str = "Labels",
        verbose_labels: bool = True,
) -> Tuple[List[Axes], List[pd.DataFrame]]:
    num_partitioners = len(partitioner_list)
    if isinstance(label_name, str):
        label_name = [label_name] * num_partitioners
    elif isinstance(label_name, List):
        pass
    else:
        raise TypeError(
            f"Label name has to be of type List[str] or str but given {type(label_name)}")
    figsize = _initialize_comparison_figsize(figsize, num_partitioners)
    fig, axes = plt.subplots(1, num_partitioners, layout='constrained', figsize=figsize)

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]
    df_list = []
    ax_list = []
    for idx, (partitioner, label_name) in enumerate(zip(partitioner_list, label_name)):
        if idx == (num_partitioners - 1):
            ax, df = plot_label_distributions(
                partitioner,
                label_name=label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                ax=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=legend,
                legend_title=legend_title,
                verbose_labels=verbose_labels,

            )
            df_list.append(df)
        else:
            ax, df = plot_label_distributions(
                partitioner,
                label_name=label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                ax=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=False,
            )
            df_list.append(df)

    for idx, ax in enumerate(axes):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(titles[idx])
    for ax in axes[1:]:
        ax.set_yticks([])

    xlabel, ylabel = _initialize_comparison_xy_labels(plot_type, partition_id_axis)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)

    fig.tight_layout()
    return fig, ax_list, df_list
