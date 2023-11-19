"""Datasets module."""
from .cifar import CIFAR10
from .mnist import MNIST
from .utils import Compose

__all__ = ("MNIST", "CIFAR10", "Compose")
