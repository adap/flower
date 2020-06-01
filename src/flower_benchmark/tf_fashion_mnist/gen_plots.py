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


def accuracy_t15() -> None:
    """Generate plots."""
    fedavg = [
        (0, 0.03759999945759773),
        (1, 0.7554000020027161),
        (2, 0.765999972820282),
        (3, 0.7705000042915344),
        (4, 0.7659000158309937),
        (5, 0.7659000158309937),
        (6, 0.7839000225067139),
        (7, 0.7954999804496765),
        (8, 0.7954999804496765),
        (9, 0.8113999962806702),
        (10, 0.8173999786376953),
        (11, 0.8166999816894531),
        (12, 0.8299999833106995),
        (13, 0.805899977684021),
        (14, 0.8126999735832214),
        (15, 0.8154000043869019),
        (16, 0.8253999948501587),
        (17, 0.8253999948501587),
        (18, 0.8169999718666077),
        (19, 0.8169999718666077),
        (20, 0.8356999754905701),
    ]
    fedfs_v0 = [
        (0, 0.03759999945759773),
        (1, 0.6930999755859375),
        (2, 0.7741000056266785),
        (3, 0.7904000282287598),
        (4, 0.829800009727478),
        (5, 0.8163999915122986),
        (6, 0.8389999866485596),
        (7, 0.833299994468689),
        (8, 0.8496999740600586),
        (9, 0.8374999761581421),
        (10, 0.8644000291824341),
        (11, 0.8478999733924866),
        (12, 0.8105999827384949),
        (13, 0.8508999943733215),
        (14, 0.8729000091552734),
        (15, 0.866599977016449),
        (16, 0.8733999729156494),
        (17, 0.862500011920929),
        (18, 0.8808000087738037),
        (19, 0.8694000244140625),
        (20, 0.8758999705314636),
    ]
    fedfs_v1 = [
        (0, 0.03759999945759773),
        (1, 0.7387999892234802),
        (2, 0.7989000082015991),
        (3, 0.8040000200271606),
        (4, 0.8212000131607056),
        (5, 0.8434000015258789),
        (6, 0.858299970626831),
        (7, 0.8651000261306763),
        (8, 0.866599977016449),
        (9, 0.8641999959945679),
        (10, 0.8418999910354614),
        (11, 0.8598999977111816),
        (12, 0.861299991607666),
        (13, 0.8784000277519226),
        (14, 0.86080002784729),
        (15, 0.8763999938964844),
        (16, 0.8407999873161316),
        (17, 0.8702999949455261),
        (18, 0.878600001335144),
        (19, 0.8791999816894531),
        (20, 0.8806999921798706),
    ]

    # Configure labels and data
    lines = [
        ("FedAvg", fedavg),
        ("FedFSv0", fedfs_v0),
        ("FedFSv1", fedfs_v1),
    ]

    # Plot
    values = [np.array([x * 100 for _, x in val]) for _, val in lines]
    labels = [label for label, _ in lines]
    line_chart(
        values,
        labels,
        "Round",
        "Accuracy",
        filename="fmnist-progress-t15",
        y_floor=60,
        y_ceil=90,
    )


def accuracy_t12() -> None:
    """Generate plots."""
    # Raw data
    fedavg = [
        (0, 0.03759999945759773),
        (1, 0.03759999945759773),
        (2, 0.03759999945759773),
        (3, 0.03759999945759773),
        (4, 0.7050999999046326),
        (5, 0.7050999999046326),
        (6, 0.7050999999046326),
        (7, 0.7465999722480774),
        (8, 0.7465999722480774),
        (9, 0.7465999722480774),
        (10, 0.7465999722480774),
        (11, 0.7465999722480774),
        (12, 0.7297000288963318),
        (13, 0.7297000288963318),
        (14, 0.7297000288963318),
        (15, 0.7297000288963318),
        (16, 0.7767999768257141),
        (17, 0.7767999768257141),
        (18, 0.7549999952316284),
        (19, 0.7549999952316284),
        (20, 0.7549999952316284),
    ]
    # fedfs_v0 = [
    #     (0, 0.03759999945759773),
    # ]
    fedfs_v1 = [
        (0, 0.03759999945759773),
        (1, 0.7074000239372253),
        (2, 0.7839999794960022),
        (3, 0.800599992275238),
        (4, 0.8098999857902527),
        (5, 0.8255000114440918),
        (6, 0.8392000198364258),
        (7, 0.8496000170707703),
        (8, 0.8640000224113464),
        (9, 0.8528000116348267),
        (10, 0.8626999855041504),
        (11, 0.8403000235557556),
        (12, 0.8288000226020813),
        (13, 0.8755999803543091),
        (14, 0.8587999939918518),
        (15, 0.8762999773025513),
        (16, 0.8531000018119812),
        (17, 0.8797000050544739),
        (18, 0.8772000074386597),
        (19, 0.8787000179290771),
        (20, 0.8792999982833862),
    ]

    # Configure labels and data
    lines = [
        ("FedAvg", fedavg),
        # ("FedFSv0", fedfs_v0),
        ("FedFSv1", fedfs_v1),
    ]

    # Plot
    values = [np.array([x * 100 for _, x in val]) for _, val in lines]
    labels = [label for label, _ in lines]
    line_chart(
        values,
        labels,
        "Round",
        "Accuracy",
        filename="fmnist-progress-t12",
        y_floor=0,
        y_ceil=100,
    )


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
        filename="fmnist-accuracy_fedavg_vs_fedfs",
    )


def wall_clock_time_fedavg_vs_fedfs() -> None:
    """Comparision of FedAvg vs FedFS."""
    bar_chart(
        y_values=[np.array([0, 1600, 1750, 2000]), np.array([650, 750, 900, 1100])],
        bar_labels=["FedAvg", "FedFS"],
        x_label="Timeout",
        x_tick_labels=["T=10", "T=20", "T=30", "T=40"],
        y_label="Completion time",
        filename="fmnist-time_fedavg_vs_fedfs",
    )


def main() -> None:
    """Call all plot functions."""
    accuracy_t12()
    accuracy_t15()
    accuracy_fedavg_vs_fedfs()
    wall_clock_time_fedavg_vs_fedfs()


if __name__ == "__main__":
    main()
