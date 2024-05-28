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
"""Label distribution bar plotting."""
from typing import Optional, Tuple, Union, Dict, Any

import datasets
import pandas as pd
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.axes import Axes


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
    plot_kwargs: Dict[str, Any],
    legend_kwargs: Dict[str, Any],

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
            handles=handles[::-1],
            labels=legend_names[::-1],
            title=legend_title,
            loc="outside center right",
            bbox_to_anchor=(1.3, 0.5),
        )

    return axis
