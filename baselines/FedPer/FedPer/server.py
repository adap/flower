"""Server strategies pipelines for FedPer."""
from flwr.server.strategy.fedavg import FedAvg

from FedPer.strategy import (
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

    pass


class AggregateBodyStrategyPipeline(
    FederatedServerPipelineStrategy, AggregateBodyStrategy, FedAvg
):
    """Aggregate body strategy pipeline."""

    pass


class DefaultStrategyPipeline(
    FederatedServerPipelineStrategy, AggregateFullStrategy, FedAvg
):
    """Default strategy pipeline."""

    pass
