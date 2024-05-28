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
"""Label distribution heatmap plotting."""
from typing import Optional, Tuple, Union, Dict, Any

import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.axes import Axes


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
    plot_kwargs: Dict[str, Any],
    legend_kwargs: Dict[str, Any],

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
