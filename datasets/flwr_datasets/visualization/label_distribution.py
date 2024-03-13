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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from partitioner import Partitioner


def plot_label_distributions(
    partitioner: Partitioner,
    label: str,
    plot_type: str,
    size_unit: str,
    legend: bool = True,
    verbose_labels: bool = True,
    partition_id_axis: str = "x",
    other_plotting_details=None,
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
        "absolute" or "percentage". "absolute" - (number of samples) there is no limit to the biggest number. It result in bars of various heights and unbounded max value on the heatmap. "percentage" - normalizes each value, so they sum up to 100%. The bars are be of the same height.
    legend: bool
        Include the legend.
    verbose_labels: bool
        Whether to use the verbose versions of the labels or keep indices representing
        them. E.g. in CIFAR10 the verbose labels are airplane, automobile, ...).
    partition_id_axis: str
        "x" or "y". The axis on which the partition_id (also known as clients or nodes) will be marked. The values are marked on the other axis.
    other_plotting_details: Any

    Returns
    -------

    Examples
    --------
    """
    if not partitioner.is_dataset_assigned():
        raise ValueError(
            "You need to assign dataset to the partitioner before using it."
        )
    if label not in partitioner.dataset.column_names:
        raise ValueError(f"The specified label: {label} is not present in the dataset.")

    partitions = []
    for partition_id in range(partitioner.num_partitions):
        partitions.append(partitioner.load_partition(partition_id))

    partition = partitions[0]
    all_labels = partition.features["label"].str2int(partition.features["label"].names)

    partition_id_to_label_absolute_size = {}
    for partition_id, partition in enumerate(partitions):
        labels_series = pd.Series(partition["label"])
        label_counts = labels_series.value_counts()
        label_counts = label_counts.reindex(all_labels, fill_value=0)
        partition_id_to_label_absolute_size[partition_id] = label_counts

    df = pd.DataFrame.from_dict(partition_id_to_label_absolute_size, orient="index")
    df.index.name = "Partition ID"
    if size_unit == "absolute":
        y_axis_label = "Count"
        pass
    elif size_unit == "percentage":
        # Divide by the total sum of samples per partition
        # Multiply by 100 (to get percentages)
        sums = df.sum(axis=1)
        df = df.div(sums, axis=0) * 100.0
        y_axis_label = "Percentage"
    else:
        raise ValueError(
            f"The size_unit can be only 'absolute' and 'percentage' but given: {size_unit}"
        )

    if plot_type == "bar":
        fig, ax = plt.subplots()
        ax = df.plot(kind="bar", stacked=True, ax=ax)
        # fig.tight_layout()
        handles, legend = ax.get_legend_handles_labels()
        # print(type(legend))
        legend_names = partition.features["label"].int2str([int(v) for v in legend])
        _ = ax.legend(
            handles[::-1],
            legend_names[::-1],
            title="Labels",
            loc="center right",
            bbox_to_anchor=(1.3, 0.5),
        )

        ax.set_ylabel(y_axis_label)
    elif plot_type == "heatmap":
        fig, ax = plt.subplots(figsize=(10, 2))

        sns.heatmap(
            df,
            ax=ax,
            cmap=sns.light_palette("seagreen", as_cmap=True),
            annot=True,
            fmt="0.2f",
            cbar=True,
        )
        ax.set_xlabel("Label")

    else:
        raise ValueError(
            f"The plot_type must be 'bar' or 'heatmap' but given: {plot_type}"
        )
    return df, (fig, ax)
