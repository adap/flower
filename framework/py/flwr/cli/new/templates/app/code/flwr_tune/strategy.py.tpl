"""$project_name: A Flower / FlowerTune app."""

from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation.

    This class behaves just like FedAvg but also tracks the communication
    costs associated with `train` over FL rounds.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()

    def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        messages = super().configure_train(server_round, arrays, config, grid)

        # Track communication costs
        self.comm_tracker.track(messages)

        return messages

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Track communication costs
        self.comm_tracker.track(replies)

        arrays, metrics = super().aggregate_train(server_round, replies)

        return arrays, metrics


class CommunicationTracker:
    """Communication costs tracker over FL rounds."""
    def __init__(self):
        self.curr_comm_cost = 0.0

    @staticmethod
    def _compute_bytes(arrays: ArrayRecord):
        return sum(len(array.data) for array in arrays.values())

    def track(self, messages: Iterable[Message]):
        record_key = next(iter(messages[0].content.array_records.keys()))
        size_bytes_list = [
            self._compute_bytes(msg.content[record_key])
            for msg in messages
        ]
        comm_cost = sum(size_bytes_list) / 1024**2

        self.curr_comm_cost += comm_cost
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        if self.curr_comm_cost > 2e5:
            log(
                WARN,
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Please consider reducing it if you plan to participate "
                "FlowerTune LLM Leaderboard.",
            )
