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
    pass


class AggregateBodyStrategyPipeline(
    FederatedServerPipelineStrategy, AggregateBodyStrategy, FedAvg
):
    pass


class DefaultStrategyPipeline(
    FederatedServerPipelineStrategy, AggregateFullStrategy, FedAvg
):
    pass
