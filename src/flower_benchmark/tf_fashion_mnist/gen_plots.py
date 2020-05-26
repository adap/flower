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
"""Generate plots for Fashion-MNIST results."""


import numpy as np

from flower_benchmark.plot import bar_chart, line_chart


def accuracy() -> None:
    """Generate plots."""
    # Raw data
    fedavg = [
        (0, 0.03759999945759773),
        (1, 0.7628999948501587),
        (2, 0.8022000193595886),
        (3, 0.832099974155426),
        (4, 0.849399983882904),
        (5, 0.8598999977111816),
        (6, 0.8655999898910522),
        (7, 0.8655999898910522),
        (8, 0.873199999332428),
        (9, 0.8780999779701233),
        (10, 0.8820000290870667),
        (11, 0.8809999823570251),
        (12, 0.8819000124931335),
        (13, 0.8827999830245972),
        (14, 0.8827000260353088),
        (15, 0.8878999948501587),
        (16, 0.8889999985694885),
        (17, 0.8888999819755554),
        (18, 0.8909000158309937),
        (19, 0.8901000022888184),
        (20, 0.8892999887466431),
        (21, 0.8919000029563904),
        (22, 0.8932999968528748),
        (23, 0.8924000263214111),
        (24, 0.8942000269889832),
        (25, 0.8944000005722046),
    ]
    fedfs = [
        (0, 0.03759999945759773),
        (1, 0.03759999945759773),
        (2, 0.7775999903678894),
        (3, 0.8069999814033508),
        (4, 0.7854999899864197),
        (5, 0.8429999947547913),
        (6, 0.8349999785423279),
        (7, 0.8658999800682068),
        (8, 0.8597999811172485),
        (9, 0.8729000091552734),
        (10, 0.8763999938964844),
        (11, 0.8822000026702881),
        (12, 0.8827000260353088),
        (13, 0.8859000205993652),
        (14, 0.8823999762535095),
        (15, 0.8877999782562256),
        (16, 0.885699987411499),
        (17, 0.8883000016212463),
        (18, 0.8860999941825867),
        (19, 0.8899999856948853),
        (20, 0.8795999884605408),
        (21, 0.8895000219345093),
        (22, 0.8881000280380249),
        (23, 0.8921999931335449),
        (24, 0.888700008392334),
        (25, 0.8930000066757202),
    ]

    # Configure labels and data
    lines = [("FedAvg", fedavg), ("FedFS", fedfs)]

    # Plot
    values = [np.array([x * 100 for _, x in val]) for _, val in lines]
    labels = [label for label, _ in lines]
    line_chart(values, labels, "Round", "Accuracy (centralized test set)")


def accuracy_fedavg_vs_fedfs() -> None:
    """Comparision of FedAvg vs FedFS."""
    bar_chart(
        y_values=[
            np.array([40.5, 85.3, 86.1, 87.2]),
            np.array([80.5, 84.3, 86.5, 89.2]),
        ],
        bar_labels=["FedAvg", "FedFS"],
        x_label="Timeout",
        x_tick_labels=["T=10", "T=20", "T=30", "T=40"],
        y_label="Final Accuracy",
        filename="accuracy_fedavg_vs_fedfs",
    )


def wall_clock_time_fedavg_vs_fedfs() -> None:
    """Comparision of FedAvg vs FedFS."""
    bar_chart(
        y_values=[np.array([0, 1600, 1750, 2000]), np.array([650, 750, 900, 1100])],
        bar_labels=["FedAvg", "FedFS"],
        x_label="Timeout",
        x_tick_labels=["T=10", "T=20", "T=30", "T=40"],
        y_label="Completion time",
        filename="wall_clock_time_fedavg_vs_fedfs",
    )


def main() -> None:
    """Call all plot functions."""
    accuracy()
    accuracy_fedavg_vs_fedfs()
    wall_clock_time_fedavg_vs_fedfs()


if __name__ == "__main__":
    main()
