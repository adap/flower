"""Server strategies pipelines for FedPer."""
from flwr.server.strategy.fedavg import FedAvg

from fedper.strategy import (
    AggregateBodyStrategy,
    AggregateFullStrategy,
    ServerInitializationStrategy,
    StoreMetricsStrategy,
    StoreSelectedClientsStrategy,
)


class FederatedServerPipelineStrategy(
    StoreSelectedClientsStrategy, StoreMetricsStrategy, ServerInitializationStrategy
):
    """Federated server pipeline strategy."""


class AggregateBodyStrategyPipeline(
    FederatedServerPipelineStrategy, AggregateBodyStrategy, FedAvg
):
    """Aggregate body strategy pipeline."""


class DefaultStrategyPipeline(
    FederatedServerPipelineStrategy, AggregateFullStrategy, FedAvg
):
    """Default strategy pipeline."""
