from flwr.server.strategy.fedavg import FedAvg

from FedPer.utils.initialization_strategy import ServerInitializationStrategy
from FedPer.utils.aggregate_body_strategy import AggregateBodyStrategy
from FedPer.utils.aggregate_full_strategy import AggregateFullStrategy
from FedPer.utils.store_metrics_strategy import StoreMetricsStrategy
from FedPer.utils.store_selected_clients_strategy import StoreSelectedClientsStrategy


class FederatedServerPipelineStrategy(
    StoreSelectedClientsStrategy,
    StoreMetricsStrategy,
    ServerInitializationStrategy
):
    pass


class AggregateBodyStrategyPipeline(
    FederatedServerPipelineStrategy,
    AggregateBodyStrategy,
    FedAvg
):
    pass


class DefaultStrategyPipeline(
    FederatedServerPipelineStrategy,
    AggregateFullStrategy,
    FedAvg
):
    pass
