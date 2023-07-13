from abc import ABC

from flwr.common.typing import SecureAggregation


class SecureAggregationHandler(ABC):
    def handle_secure_aggregation(self, sa: SecureAggregation) -> SecureAggregation:
        ...
