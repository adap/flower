"""
Feature Election for Flower

Federated feature selection framework for tabular datasets.
"""

__version__ = "0.9.0"
__author__ = "Ioannis Christofilogiannis"

from .client_app import FeatureElectionClient, app as client_app
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
    "FeatureElectionClient",
    "FeatureSelector",
    "create_synthetic_dataset",
    "load_client_data",
    "load_custom_dataset",
    "prepare_federated_dataset",
    "client_app",
    "server_app",
]
