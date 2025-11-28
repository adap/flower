"""
Feature Election for Flower

Federated feature selection framework for tabular datasets.
Uses the Flower Message API for communication.
"""

__version__ = "1.0.0"
__author__ = "Ioannis Christofilogiannis"

from .client_app import app as client_app
from .feature_election_utils import FeatureSelector
from .server_app import app as server_app
from .strategy import FeatureElectionStrategy
from .task import (
    create_synthetic_dataset,
    load_client_data,
    load_custom_dataset,
    prepare_federated_dataset,
)

__all__ = [
    "FeatureElectionStrategy",
    "FeatureSelector",
    "create_synthetic_dataset",
    "load_client_data",
    "load_custom_dataset",
    "prepare_federated_dataset",
    "client_app",
    "server_app",
]
