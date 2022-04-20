from re import S
import pytorch
from torch import nn


def get_model():
    net = nn.Sequetial(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Dropout(p=0.25),
        nn.Flatten(),
        nn.Linear(in_features=9216, out_features=128),
        nn.Dropout(p=0.50),
        nn.Linear(in_features=128, out_features=62),
    )
