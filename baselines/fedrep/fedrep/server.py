"""Server strategies pipelines for FedRep."""

from flwr.server.strategy.fedavg import FedAvg

from fedrep.strategy import (
    AggregateBodyStrategy,
    AggregateFullStrategy,
    ServerInitializationStrategy,
)


class InitializationStrategyPipeline(ServerInitializationStrategy):
    """Initialization strategy pipeline."""


class AggregateBodyStrategyPipeline(
    InitializationStrategyPipeline, AggregateBodyStrategy, FedAvg
):
    """Aggregate body strategy pipeline."""


class DefaultStrategyPipeline(
    InitializationStrategyPipeline, AggregateFullStrategy, FedAvg
):
    """Default strategy pipeline."""
