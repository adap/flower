from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import flwr as fl


def create_broadcast_messages(
    driver: fl.Driver,
    record: fl.RecordSet,
    message_type: fl.MessageType,
    node_ids: list[int],
    group_id: str,
) -> list[fl.Message]:

    messages = []
    for node_id in node_ids:  # one message for each node
        # Copy non-parameters records to avoid modifying the original record
        new_record = fl.RecordSet(
            parameters_records=record.parameters_records,
            metrics_records={k: v.copy() for k, v in record.metrics_records.items()},
            configs_records={k: v.copy() for k, v in record.configs_records.items()},
        )
        message = driver.create_message(
            content=new_record,
            message_type=message_type,  # target method in ClientApp
            dst_node_id=node_id,
            group_id=group_id,
        )
        messages.append(message)
    return messages


class BaseAggregator:

    def __call__(self, messages: list[fl.Message]) -> fl.RecordSet:
        return self.aggregate(messages)

    @abstractmethod
    def aggregate(self, messages: list[fl.Message]) -> fl.RecordSet: ...


class SequentialAggregator(BaseAggregator):

    def __init__(
        self,
        aggregators: list[BaseAggregator],
    ) -> None: ...


class ParametersAggregator(BaseAggregator):

    def __init__(
        self,
        record_key: str | None = None,
        weight_factor_key: Callable[[fl.RecordSet], float | int] | None = None,
        aggregate_key: list[str] | str = "*",
        reduction: str = "mean",
    ) -> None: ...

    def aggregate(self, messages: list[fl.Message]) -> fl.RecordSet: ...


class MetricsAggregator(BaseAggregator):

    def __init__(
        self,
        record_key: str | None = None,
        weight_factor_key: Callable[[fl.RecordSet], float | int] | str | None = None,
        aggregate_key: list[str] | str = "*",
        reduction: str = "mean",
    ) -> None: ...

    def aggregate(self, messages: list[fl.Message]) -> fl.RecordSet: ...
