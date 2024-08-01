"""Server strategies pipelines for FedRep."""

from flwr.server.strategy.fedavg import FedAvg

from fedrep.strategy import AggregateFullStrategy, ServerInitializationStrategy


class InitializationStrategyPipeline(ServerInitializationStrategy):
    """Initialization strategy pipeline."""


class DefaultStrategyPipeline(
    InitializationStrategyPipeline, AggregateFullStrategy, FedAvg
):
    """Default strategy pipeline."""
