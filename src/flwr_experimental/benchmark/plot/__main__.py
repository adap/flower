# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Provides plotting functions."""

import random

import numpy as np

from . import bar_chart, line_chart, single_bar_chart

e1 = np.array([random.uniform(0, i / 5) + 50 for i in range(100)])
e5 = np.array([random.uniform(0, i / 5) + 25 for i in range(100)])
e10 = np.array([random.uniform(0, i / 5) + 0 for i in range(100)])

line_chart(
    [e1, e5, e10], ["E=1", "E=5", "E=10"], "Rounds", "Test Accuracy",
)

bar_chart(
    y_values=[np.array([51.1, 52.3, 10.0]), np.array([59.3, 73.5, 80.0])],
    bar_labels=["GPU-fast", "GPU-mixed"],
    x_label="Local Epochs",
    x_tick_labels=["E1", "E2", "E3"],
    y_label="Training time",
)

single_bar_chart(
    np.array([50.0, 70.0, 80.0]),
    ["10", "100", "1000"],
    "Number of Clients",
    "Test Accuracy",
)
