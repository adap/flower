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


import warnings
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.colors as mcolors
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flwr_datasets.metrics import compute_counts
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.visualization.bar_plot import _plot_bar
from flwr_datasets.visualization.heatmap_plot import _plot_heatmap
from flwr_datasets.visualization.utils import _validate_parameters

# pylint: disable=too-many-arguments,too-many-locals


def plot_label_distributions(
    partitioner: Partitioner,
    label_name: str,
    plot_type: str = "bar",
    size_unit: str = "absolute",
    max_num_partitions: Optional[int] = None,
    partition_id_axis: str = "x",
    axis: Optional[Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Per Partition Label Distribution",
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    legend: bool = False,
    legend_title: Optional[str] = None,
    verbose_labels: bool = True,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    legend_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Figure, Axes, pd.DataFrame]:
    """Plot the label distribution of the partitions.

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
    axis : Optional[Axes]
        Matplotlib Axes object to plot on.
    figsize : Optional[Tuple[float, float]]
        Size of the figure.
    title : str
        Title of the plot.
    cmap : Optional[Union[str, mcolors.Colormap]]
        Colormap for determining the colorspace of the plot.
    legend : bool
        Include the legend.
    legend_title : Optional[str]
        Title for the legend. If None, the defaults will be takes based on the type of
        plot.
    verbose_labels : bool
        Whether to use verbose versions of the labels.
    plot_kwargs: Optional[Dict[str, Any]]
        Any key value pair that can be passed to a plot function that are not supported
        directly. In case of the parameter doubling (e.g. specifying cmap here too) the
        chosen value will be taken from the explicit arguments (e.g. cmap specified as
        an argument to this function not the value in this dictionary).
    legend_kwargs: Optional[Dict[str, Any]]
        Any key value pair that can be passed to a figure.legend in case of bar plot or
        cbar_kws in case of heatmap that are not supported directly. In case of the
        parameter doubling (e.g. specifying legend_title here too) the
        chosen value will be taken from the explicit arguments (e.g. legend_title
        specified as an argument to this function not the value in this dictionary).

    Returns
    -------
    fig : Figure
        The figure object.
    axis : Axes
        The Axes object with the plot.
    dataframe : pd.DataFrame
        The DataFrame where each row represents the partition id and each column
        represents the class.

    Examples
    --------
    Visualize the label distribution resulting from DirichletPartitioner.

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> figure, axis, dataframe = plot_label_distributions(
    >>>     partitioner=partitioner,
    >>>     label_name="label",
    >>>     legend=True,
    >>>     verbose_labels=True,
    >>> )

    Alternatively you can visualize each partition in terms of fraction of the data
    available on that partition instead of the absolute count

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> figure, axis, dataframe = plot_label_distributions(
    >>>     partitioner=partitioner,
    >>>     label_name="label",
    >>>     size_unit="percent",
    >>>     legend=True,
    >>>     verbose_labels=True,
    >>> )
    >>>

    You can also visualize the data as a heatmap by changing the `plot_type` from
    default "bar" to "heatmap"

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> figure, axis, dataframe = plot_label_distributions(
    >>>     partitioner=partitioner,
    >>>     label_name="label",
    >>>     size_unit="percent",
    >>>     plot_type="heatmap",
    >>>     legend=True,
    >>>     plot_kwargs={"annot": True},
    >>> )

    You can also visualize the returned DataFrame in Jupyter Notebook
    >>> dataframe.style.background_gradient(axis=None)
    """
    _validate_parameters(plot_type, size_unit, partition_id_axis)

    if label_name not in partitioner.dataset.column_names:
        raise ValueError(
            f"The specified 'label_name': '{label_name}' is not present in the "
            f"dataset. The dataset contains columns {partitioner.dataset.column_names}."
        )

    if max_num_partitions is None:
        max_num_partitions = partitioner.num_partitions
    else:
        max_num_partitions = min(max_num_partitions, partitioner.num_partitions)
    assert isinstance(max_num_partitions, int)
    partitions = [partitioner.load_partition(i) for i in range(max_num_partitions)]

    partition = partitions[0]
    try:
        # Unique labels are needed to represent the correct count of each class
        # (some of the classes can have zero samples that's why this
        # adjustment is needed)
        unique_labels = partition.features[label_name].str2int(
            partition.features[label_name].names
        )
    except AttributeError:  # If the label_name is not formally a Label
        unique_labels = partitioner.dataset.unique(label_name)

    partition_id_to_label_absolute_size = {
        pid: compute_counts(partition[label_name], unique_labels)
        for pid, partition in enumerate(partitions)
    }

    dataframe = pd.DataFrame.from_dict(
        partition_id_to_label_absolute_size, orient="index"
    )
    dataframe.index.name = "Partition ID"

    if size_unit == "percent":
        dataframe = dataframe.div(dataframe.sum(axis=1), axis=0) * 100.0

    if verbose_labels:
        # Adjust the column name values of the dataframe
        # (these values are used for as labels in bar plot and columns/rows ticks
        # in heatmap)
        current_labels = dataframe.columns
        try:
            legend_names = partition.features[label_name].int2str(
                [int(v) for v in current_labels]
            )
            dataframe.columns = legend_names
        except AttributeError:
            warnings.warn(
                "The verbose label names can not be established. "
                "The column specified by 'label_name' needs to be of type "
                "'ClassLabel'",
                stacklevel=1,
            )

    if plot_type == "bar":
        axis = _plot_bar(
            dataframe,
            axis,
            figsize,
            title,
            cmap,
            partition_id_axis,
            size_unit,
            legend,
            legend_title,
            plot_kwargs,
            legend_kwargs,
        )
    elif plot_type == "heatmap":
        axis = _plot_heatmap(
            dataframe,
            axis,
            figsize,
            title,
            cmap,
            partition_id_axis,
            size_unit,
            legend,
            legend_title,
            plot_kwargs,
            legend_kwargs,
        )
    assert axis is not None
    return axis.figure, axis, dataframe
