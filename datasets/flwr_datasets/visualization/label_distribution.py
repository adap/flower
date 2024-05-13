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
from typing import Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from metrics import compute_counts
from partitioner import Partitioner
import seaborn as sns

# sns.set_theme()
import numpy as np

sns.reset_orig()



def plot_label_distributions(
        # Flower Datasets' specific parameters
        partitioner: Partitioner,
        label: str,
        plot_type: str,
        size_unit: str,
        num_partitions: Optional[int] = None,
        partition_id_axis: str = "x",
        # Plotting specific parameters
        ax: Optional[matplotlib.axes.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Per Partition Label Distribution",
        colormap=None,
        legend: bool = True,
        legend_name: str = "Labels",
        verbose_labels: bool = True,
        **plot_kwargs,
):
    """Plot the label distribution of the.

    Parameters
    ----------
    partitioner: Partitioner
        Partitioner with an assigned dataset.
    label: str
        Label based on which the plot will be created.
    plot_type: str
        Type of plot, either "bar" or "heatmap".
    size_unit: str
         "absolute" or "percent". "absolute" - (number of samples) there is no limit
        to the biggest number. It results in bars of various heights and unbounded max
        value on the heatmap. "percent" - normalizes each value, so they sum up to
        100%. The bars are be of the same height.
    num_partitions: Optional[int]
        The number of partitions that will be used. If left None, then all the
        partitions will be used.
    legend: bool
        Include the legend.
    verbose_labels: bool
        Whether to use the verbose versions of the labels or keep indices representing
        them. E.g. in CIFAR10 the verbose labels are airplane, automobile, ...).
    partition_id_axis: str
        "x" or "y". The axis on which the partition_id (also known as clients or
        nodes) will be marked. The values are marked on the other axis.

    Returns
    -------

    Examples
    --------
    """
    if label not in partitioner.dataset.column_names:
        raise ValueError(
            f"The specified label: '{label}' is not present in the dataset.")

    # Load all partitions
    partitions = []
    if num_partitions is None:
        num_partitions = partitioner.num_partitions
    for partition_id in range(num_partitions):
        partitions.append(partitioner.load_partition(partition_id))

    # Infer the label information based on any (part) of the dataset
    partition = partitions[0]
    unique_labels = partition.features[label].str2int(partition.features[label].names)

    pid_to_label_absolute_size = {}
    for pid, partition in enumerate(partitions):
        pid_to_label_absolute_size[pid] = compute_counts(partition[label],
                                                         unique_labels)

    df = pd.DataFrame.from_dict(pid_to_label_absolute_size, orient="index")
    df.index.name = "Partition ID"

    # Adjust the data based on the size_unit
    if size_unit == "absolute":
        # No operation in case of absolute
        pass
    elif size_unit == "percent":
        # Divide by the total sum of samples per partition
        # Multiply by 100 (to get percentages)
        sums = df.sum(axis=1)
        df = df.div(sums, axis=0) * 100.0
    else:
        raise ValueError(
            f"The size_unit can be only 'absolute' and 'percentage' but given: "
            f"{size_unit}"
        )

    # Transpose the data if the partition_id_axis == "y"
    if partition_id_axis == "x" and plot_type == "heatmap":
        df = df.T

    # Figure out label naming
    xlabel, ylabel = _initialize_xy_labels(plot_type, size_unit, partition_id_axis)

    cbar_title = _initialize_cbar_title(plot_type, size_unit)

    figsize = _initialize_figsize(figsize, plot_type, partition_id_axis, num_partitions)

    if plot_type == "bar":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        kind = "bar" if partition_id_axis == "x" else "barh"
        ax = df.plot(kind=kind, stacked=True, ax=ax, title=title, legend=False, **plot_kwargs)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            handles, legend = ax.get_legend_handles_labels()
            if verbose_labels:
                legend_names = partition.features[label].int2str([int(v) for v in legend])
            else:
                legend_names = legend
            _ = fig.legend(
                handles[::-1],
                legend_names[::-1],
                title=legend_name,
                loc="outside center right",
                bbox_to_anchor=(1.3, 0.5),
            )
        # fig.tight_layout()
        # fig.subplots_adjust()


    elif plot_type == "heatmap":
        if colormap is None:
            colormap = sns.light_palette("seagreen", as_cmap=True)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if size_unit == "absolute":
            fmt = ",d"
        elif size_unit == "percent":
            fmt = "0.2f"
        else:
            raise ValueError(
                f"The size_unit can be only 'absolute' and 'percentage' but given: "
                f"{size_unit}"
            )
        sns.heatmap(
            df,
            ax=ax,
            cmap=colormap,
            annot=True,
            fmt=fmt,
            cbar=legend,
            cbar_kws={'label': cbar_title}
        )
        # ax.set_xlabel needs to be below the sns.heatmap
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        ax.set_title(title)
    else:
        raise ValueError(
            f"The plot_type must be 'bar' or 'heatmap' but given: {plot_type}"
        )
    return ax, df


def _initialize_xy_labels(plot_type, size_unit, partition_id_axis):
    xlabel = None
    ylabel = None
    if plot_type == "bar":
        xlabel = "Partition ID"
        if size_unit == "absolute":
            ylabel = "Count"
        elif size_unit == "percent":
            ylabel = "Percent %"
        else:
            raise ValueError(
                f"The size_unit can be only 'absolute' and 'percentage' but given: "
                f"{size_unit}"
            )
    elif plot_type == "heatmap":
        xlabel = "Partition ID"
        ylabel = "Label"
    else:
        raise ValueError(
            f"The plot_type must be 'bar' or 'heatmap' but given: {plot_type}"
        )
    # Flip the labels if partition_id_axis == "y"
    if partition_id_axis == "x":
        pass
    elif partition_id_axis == "y":
        temp = xlabel
        xlabel = ylabel
        ylabel = temp
    else:
        raise ValueError(
            f"The partition_id_axis needs to be 'x' or 'y' but '{partition_id_axis}' was given.")
    return xlabel, ylabel


def _initialize_cbar_title(plot_type, size_unit):
    cbar_title = None
    if plot_type == "heatmap":
        if size_unit == "absolute":
            cbar_title = "Count"
        elif size_unit == "percent":
            cbar_title = "Percent %"
        else:
            raise ValueError(
                f"The size_unit can be only 'absolute' and 'percentage' but given: "
                f"{size_unit}"
            )
    return cbar_title


def _initialize_figsize(figsize: Tuple[float, float], plot_type: str, partition_id_axis: str, num_partitions: int, num_labels=10) -> Tuple[float, float]:
    # todo: num_labels is something that will need to be incorporated
    if figsize is not None:
        return figsize
    if plot_type == "bar":
        # Other good heuristic is log2 (log and log10 seems to produce too narrow plots)
        if partition_id_axis == "x":
            figsize = (6.4, 4.8)
        elif partition_id_axis == "y":
            figsize = (6.4, np.sqrt(num_partitions))
        else:
            raise ValueError(
                f"The partition_id_axis needs to be 'x' or 'y' but '{partition_id_axis}' was given.")
    elif plot_type == "heatmap":
        if partition_id_axis == "x":
            # The np.sqrt(num_partitions) is too small even for 20 partitions
            # the numbers start to overlap
            # 2 is reasonable coef but probably in this case manual adjustment
            # will be needed
            figsize = (3 * np.sqrt(num_partitions), 6.4)
        elif partition_id_axis == "y":
            figsize = (6.4, np.sqrt(num_partitions))
        raise ValueError(
            f"The partition_id_axis needs to be 'x' or 'y' but '{partition_id_axis}' was given.")
    else:
        raise ValueError("Plot type can be only bar and heatmap.")
    return figsize


