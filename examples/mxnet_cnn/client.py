"""Flower client example using PyTorch for CIFAR-10 image classification."""

import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import mxnet

from . import mxnet_cnn