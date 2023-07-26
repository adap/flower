from flwr.server.strategy.fedavg import FedAvg

from fedpfl.federated_learning.strategy.aggregate_body_strategy import (
    AggregateBodyStrategy,
)
from fedpfl.federated_learning.strategy.aggregate_full_strategy import (
    AggregateFullStrategy,
)
from fedpfl.federated_learning.strategy.aggregate_head_strategy import (
    AggregateHeadStrategy,
)
from fedpfl.federated_learning.strategy.aggregate_hybrid_babulg_dual_strategy import (
    AggregateHybridBABULGStrategy,
)
from fedpfl.federated_learning.strategy.initialization_strategy import (
    ServerInitializationStrategy,
)
from fedpfl.federated_learning.strategy.store_metrics_strategy import (
    StoreMetricsStrategy,
)
from fedpfl.federated_learning.strategy.store_selected_clients_strategy import (
    StoreSelectedClientsStrategy,
)


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


class AggregateHeadStrategyPipeline(
    FederatedServerPipelineStrategy,
    AggregateHeadStrategy,
    FedAvg
):
    pass


class AggregateHybridBABULGStrategyPipeline(
    FederatedServerPipelineStrategy,
    AggregateHybridBABULGStrategy,
    FedAvg
):
    pass


class DefaultStrategyPipeline(
    FederatedServerPipelineStrategy,
    AggregateFullStrategy,
    FedAvg
):
    pass
