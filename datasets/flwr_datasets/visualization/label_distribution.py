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

sns.set_theme()


def plot_label_distributions(
        # Flower Datasets' specific parameters
        partitioner: Partitioner,
        label: str,
        plot_type: str,
        size_unit: str,
        partition_id_axis: str = "x",
        # Plotting specific parameters
        ax: matplotlib.axes.Axes = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "",
        colormap=None,
        legend: bool = True,
        verbose_labels: bool = False,
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
    for partition_id in range(partitioner.num_partitions):
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

    # Perform 'size_unit'-specific operations
    if size_unit == "absolute":
        y_axis_label = "Count"
        pass
    elif size_unit == "percent":
        # Divide by the total sum of samples per partition
        # Multiply by 100 (to get percentages)
        sums = df.sum(axis=1)
        df = df.div(sums, axis=0) * 100.0
        y_axis_label = "Percent %"
    else:
        raise ValueError(
            f"The size_unit can be only 'absolute' and 'percentage' but given: "
            f"{size_unit}"
        )

    if plot_type == "bar":
        fig, ax = plt.subplots()
        fig.tight_layout()
        ax = df.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel(y_axis_label)
        if label:
            handles, legend = ax.get_legend_handles_labels()
            legend_names = partition.features["label"].int2str([int(v) for v in legend])
            _ = ax.legend(
                handles[::-1],
                legend_names[::-1],
                title="Labels",
                loc="center right",
                bbox_to_anchor=(1.3, 0.5),
            )

    elif plot_type == "heatmap":
        if colormap is None:
            colormap = sns.light_palette("seagreen", as_cmap=True)
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap(
            df,
            ax=ax,
            cmap=colormap,
            annot=True,
            fmt="0.2f",
            cbar=legend,
        )
        # ax.set_xlabel needs to be below the sns.heatmap
        ax.set_xlabel("Label")

    else:
        raise ValueError(
            f"The plot_type must be 'bar' or 'heatmap' but given: {plot_type}"
        )
    return ax, df
