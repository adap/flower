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
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import datasets
from flwr_datasets.metrics import compute_counts
from flwr_datasets.partitioner import Partitioner

# pylint: disable=too-many-arguments,too-many-locals
# Constants for plot types and size units
PLOT_TYPES = ("bar", "heatmap")
SIZE_UNITS = ("absolute", "percent")
AXIS_TYPES = ("x", "y")


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
    legend_title: str = "Labels",
    verbose_labels: bool = True,
    **plot_kwargs: Dict[str, Any],
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
        Colormap for the heatmap.
    legend : bool
        Include the legend.
    legend_title : str
        Name for the legend.
    verbose_labels : bool
        Whether to use verbose versions of the labels.

    Returns
    -------
    fig : Figure
        The figure object.
    axis : Axes
        The Axes object with the plot.
    dataframe : pd.DataFrame
        The DataFrame used for plotting.

    Examples
    --------
    Visualize the label distribution resulting from DirichletPartitioner.

    >> from flwr_datasets import FederatedDataset
    >> from flwr_datasets.partitioner import DirichletPartitioner
    >> from flwr_datasets.visualization import compare_label_distribution
    >>
    >> fds = FederatedDataset(
    >>     dataset="cifar10",
    >>     partitioners={
    >>         "train": DirichletPartitioner(
    >>             num_partitions=20,
    >>             partition_by="label",
    >>             alpha=0.3,
    >>             min_partition_size=0,
    >>         ),
    >>     },
    >> )
    >> partitioner = fds.partitioners["train"]
    >> figure, axis, dataframe = plot_label_distributions(
    >>     partitioner=partitioner,
    >>     label_name="label"
    >> )

    Alternatively you can visualize each partition in terms of fraction of the data
    available on that partition instead of the absolute count

    >> from flwr_datasets import FederatedDataset
    >> from flwr_datasets.partitioner import DirichletPartitioner
    >> from flwr_datasets.visualization import compare_label_distribution
    >>
    >> fds = FederatedDataset(
    >>     dataset="cifar10",
    >>     partitioners={
    >>         "train": DirichletPartitioner(
    >>             num_partitions=20,
    >>             partition_by="label",
    >>             alpha=0.3,
    >>             min_partition_size=0,
    >>         ),
    >>     },
    >> )
    >> partitioner = fds.partitioners["train"]
    >> figure, axis, dataframe = plot_label_distributions(
    >>     partitioner=partitioner,
    >>     label_name="label"
    >>     size_unit="percent",
    >> )
    >>

    You can also visualize the data as a heatmap by changing the `plot_type` from
    default "bar" to "heatmap"

    >> from flwr_datasets import FederatedDataset
    >> from flwr_datasets.partitioner import DirichletPartitioner
    >> from flwr_datasets.visualization import compare_label_distribution
    >>
    >> fds = FederatedDataset(
    >>     dataset="cifar10",
    >>     partitioners={
    >>         "train": DirichletPartitioner(
    >>             num_partitions=20,
    >>             partition_by="label",
    >>             alpha=0.3,
    >>             min_partition_size=0,
    >>         ),
    >>     },
    >> )
    >> partitioner = fds.partitioners["train"]
    >> figure, axis, dataframe = plot_label_distributions(
    >>     partitioner=partitioner,
    >>     label_name="label"
    >>     size_unit="percent",
    >>     plot_type="heatmap",
    >>     annot=True,
    >> )

    You can also visualize the returned DataFrame in Jupyter Notebook
    >> df.style.background_gradient(axis=None)
    """
    _validate_parameters(plot_type, size_unit, partition_id_axis)

    if label_name not in partitioner.dataset.column_names:
        raise ValueError(
            f"The specified 'label_name': '{label_name}' is not present in the "
            f"dataset."
        )

    if max_num_partitions is None:
        max_num_partitions = partitioner.num_partitions
    else:
        max_num_partitions = min(max_num_partitions, partitioner.num_partitions)
    assert isinstance(max_num_partitions, int)
    partitions = [partitioner.load_partition(i) for i in range(max_num_partitions)]

    partition = partitions[0]
    try:
        unique_labels = partition.features[label_name].str2int(
            partition.features[label_name].names
        )
    except AttributeError:  # If the label_name is not formally a Label
        unique_labels = partitioner.dataset.unique(label_name)
    num_labels = len(unique_labels)

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

    if partition_id_axis == "x" and plot_type == "heatmap":
        dataframe = dataframe.T

    xlabel, ylabel = _initialize_xy_labels(plot_type, size_unit, partition_id_axis)
    cbar_title = _initialize_cbar_title(plot_type, size_unit)
    figsize = _initialize_figsize(
        figsize, plot_type, partition_id_axis, max_num_partitions, num_labels
    )

    axis = _plot_data(
        dataframe,
        plot_type,
        axis,
        figsize,
        title,
        cmap,
        xlabel,
        ylabel,
        cbar_title,
        legend,
        verbose_labels,
        legend_title,
        partition,
        label_name,
        **plot_kwargs,
    )

    return axis.figure, axis, dataframe


def _initialize_xy_labels(
    plot_type: str, size_unit: str, partition_id_axis: str
) -> Tuple[str, str]:
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


def _validate_parameters(
    plot_type: str, size_unit: str, partition_id_axis: str
) -> None:
    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}."
        )
    if size_unit not in SIZE_UNITS:
        raise ValueError(
            f"Invalid size_unit: {size_unit}. Must be one of {SIZE_UNITS}."
        )
    if partition_id_axis not in AXIS_TYPES:
        raise ValueError(
            f"Invalid partition_id_axis: {partition_id_axis}. Must be 'x' or 'y'."
        )


def _initialize_cbar_title(plot_type: str, size_unit: str) -> Optional[str]:
    if plot_type == "heatmap":
        return "Count" if size_unit == "absolute" else "Percent %"
    return None


def _initialize_figsize(
    figsize: Optional[Tuple[float, float]],
    plot_type: str,
    partition_id_axis: str,
    num_partitions: int,
    num_labels: int,
) -> Tuple[float, float]:
    if figsize is not None:
        if not isinstance(figsize, tuple):
            raise TypeError(
                f"'figsize' should of type 'tuple' but given: {type(figsize)}"
            )
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
    else:
        raise ValueError(
            f"The type of plot must be 'bar' or 'heatmap' but given: {plot_type}"
        )
    assert figsize is not None
    return figsize


def _plot_data(
    dataframe: pd.DataFrame,
    plot_type: str,
    axis: Optional[Axes],
    figsize: Tuple[float, float],
    title: str,
    colormap: Optional[Union[str, mcolors.Colormap]],
    xlabel: str,
    ylabel: str,
    cbar_title: Optional[str],
    legend: bool,
    verbose_labels: bool,
    legend_title: str,
    partition: datasets.Dataset,
    label_name: str,
    **plot_kwargs: Dict[str, Any],
) -> Axes:
    if plot_type == "bar":
        return _plot_bar(
            dataframe,
            axis,
            figsize,
            title,
            colormap,
            xlabel,
            ylabel,
            legend,
            legend_title,
            verbose_labels,
            partition,
            label_name,
            **plot_kwargs,
        )
    if plot_type == "heatmap":
        return _plot_heatmap(
            dataframe,
            axis,
            figsize,
            title,
            colormap,
            xlabel,
            ylabel,
            cbar_title,
            legend,
            **plot_kwargs,
        )
    raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'bar' or 'heatmap'.")


def _plot_bar(
    dataframe: pd.DataFrame,
    axis: Optional[Axes],
    figsize: Tuple[float, float],
    title: str,
    colormap: Optional[Union[str, mcolors.Colormap]],
    xlabel: str,
    ylabel: str,
    legend: bool,
    legend_title: str,
    verbose_labels: bool,
    partition: datasets.Dataset,
    label_name: str,
    **plot_kwargs: Dict[str, Any],
) -> Axes:
    if colormap is None:
        colormap = "RdYlGn"
    if axis is None:
        _, axis = plt.subplots(figsize=figsize)

    kind = "bar" if xlabel == "Partition ID" else "barh"
    axis = dataframe.plot(
        kind=kind,
        stacked=True,
        ax=axis,
        title=title,
        legend=False,
        colormap=colormap,
        rot=0,
        **plot_kwargs,
    )

    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)

    xticklabels = axis.get_xticklabels()
    if len(xticklabels) > 20:
        # Make every other xtick label not visible
        for i, label in enumerate(xticklabels):
            if i % 2 == 1:
                label.set_visible(False)

    if legend:
        handles, legend_labels = axis.get_legend_handles_labels()
        if verbose_labels:
            try:
                legend_names = partition.features[label_name].int2str(
                    [int(v) for v in legend_labels]
                )
            except AttributeError:
                legend_names = legend_labels
        else:
            legend_names = legend_labels

        _ = axis.figure.legend(
            handles[::-1],
            legend_names[::-1],
            title=legend_title,
            loc="outside center right",
            bbox_to_anchor=(1.3, 0.5),
        )

    return axis


def _plot_heatmap(
    dataframe: pd.DataFrame,
    axis: Optional[Axes],
    figsize: Tuple[float, float],
    title: str,
    colormap: Optional[Union[str, mcolors.Colormap]],
    xlabel: str,
    ylabel: str,
    cbar_title: Optional[str],
    legend: bool,
    **plot_kwargs: Dict[str, Any],
) -> Axes:
    if colormap is None:
        colormap = sns.light_palette("seagreen", as_cmap=True)
    if axis is None:
        _, axis = plt.subplots(figsize=figsize)

    fmt = ",d" if "absolute" in dataframe.columns else "0.2f"
    sns.heatmap(
        dataframe,
        ax=axis,
        cmap=colormap,
        fmt=fmt,
        cbar=legend,
        cbar_kws={"label": cbar_title},
        **plot_kwargs,
    )

    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)

    axis.set_title(title)
    return axis


def _initialize_comparison_figsize(
    figsize: Optional[Tuple[float, float]], num_partitioners: int
) -> Tuple[float, float]:
    if figsize is not None:
        return figsize
    x_value = 4 + (num_partitioners - 1) * 2
    y_value = 4.8
    figsize = (x_value, y_value)
    return figsize


def _initialize_comparison_xy_labels(
    plot_type: str, partition_id_axis: str
) -> Tuple[str, str]:
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
) -> Tuple[Figure, List[Axes], List[pd.DataFrame]]:
    """Compare the label distribution across multiple partitioners.

    Parameters
    ----------
    partitioner_list : List[Partitioner]
        List of partitioners to be compared.
    label_name : Union[str, List[str]]
        Column name or list of column names identifying labels for each partitioner.
    plot_type : str, default "bar"
        Type of plot, either "bar" or "heatmap".
    size_unit : str, default "percent"
        "absolute" for raw counts, or "percent" to normalize values to 100%.
    max_num_partitions : Optional[int], default 30
        Maximum number of partitions to include in the plot. If None, all partitions
        are included.
    partition_id_axis : str, default "y"
        Axis on which the partition IDs will be marked, either "x" or "y".
    figsize : Optional[Tuple[float, float]], default None
        Size of the figure. If None, a default size is calculated.
    subtitle : str, default "Comparison of Per Partition Label Distribution"
        Subtitle for the figure.
    titles : Optional[List[str]], default None
        Titles for each subplot. If None, no titles are set.
    cmap : Optional[Union[str, mcolors.Colormap]], default None
        Colormap for the heatmap.
    legend : bool, default False
        Whether to include a legend.
    legend_title : str, default "Labels"
        Title for the legend.
    verbose_labels : bool, default True
        Whether to use verbose versions of the labels.

    Returns
    -------
    fig : Figure
        The figure object containing the plots.
    axes_list : List[Axes]
        List of Axes objects for the plots.
    dataframe_list : List[pd.DataFrame]
        List of DataFrames used for each plot.

    Examples
    --------
    Compare the difference of using different alpha (concentration) parameters in
    DirichletPartitioner.

    >> from flwr_datasets import FederatedDataset
    >> from flwr_datasets.partitioner import DirichletPartitioner
    >> from flwr_datasets.visualization import compare_label_distribution
    >>
    >> partitioner_list = []
    >> alpha_list = [10_000.0, 100.0, 1.0, 0.1, 0.01, 0.00001]
    >> for alpha in alpha_list:
    >>     fds = FederatedDataset(
    >>         dataset="cifar10",
    >>         partitioners={
    >>             "train": DirichletPartitioner(
    >>                 num_partitions=20,
    >>                 partition_by="label",
    >>                 alpha=alpha,
    >>                 min_partition_size=0,
    >>             ),
    >>         },
    >>     )
    >>     partitioner_list.append(fds.partitioners["train"])
    >> fig, axes, dataframe_list = compare_label_distribution(
    >>     partitioner_list=partitioner_list,
    >>     label_name="label",
    >>     titles=[f"Concentration = {alpha}" for alpha in alpha_list],
    >> )
    """
    num_partitioners = len(partitioner_list)
    if isinstance(label_name, str):
        label_name = [label_name] * num_partitioners
    elif isinstance(label_name, List):
        pass
    else:
        raise TypeError(
            f"Label name has to be of type List[str] or str but given "
            f"{type(label_name)}"
        )
    figsize = _initialize_comparison_figsize(figsize, num_partitioners)
    fig, axes = plt.subplots(1, num_partitioners, layout="constrained", figsize=figsize)

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]
    dataframe_list = []
    for idx, (partitioner, single_label_name) in enumerate(
        zip(partitioner_list, label_name)
    ):
        if idx == (num_partitioners - 1):
            *_, dataframe = plot_label_distributions(
                partitioner,
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=legend,
                legend_title=legend_title,
                verbose_labels=verbose_labels,
            )
            dataframe_list.append(dataframe)
        else:
            *_, dataframe = plot_label_distributions(
                partitioner,
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=False,
            )
            dataframe_list.append(dataframe)

    for idx, axis in enumerate(axes):
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(titles[idx])
    for axis in axes[1:]:
        axis.set_yticks([])

    xlabel, ylabel = _initialize_comparison_xy_labels(plot_type, partition_id_axis)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)

    fig.tight_layout()
    return fig, axes, dataframe_list
