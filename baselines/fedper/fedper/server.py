"""Server strategies pipelines for FedPer."""

from flwr.server.strategy.fedavg import FedAvg

from fedper.strategy import (
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
