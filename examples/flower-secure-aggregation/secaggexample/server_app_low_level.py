"""secaggexample: A Flower with SecAgg+ app."""

import time
from collections.abc import Iterable
from logging import DEBUG, INFO

from flwr.common import Context, Message, RecordSet, log
from flwr.common.constant import MessageType
from flwr.common.logger import update_console_handler
from flwr.server import Driver, ServerApp
from flwr.server.workflow import SecAggPlusAggregator
from flwr.server.workflow.secure_aggregation.secaggplus_aggregator import (
    SecAggPlusAggregatorState,
)

# Flower ServerApp
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Show debug logs
    update_console_handler(DEBUG, True, True)

    # Sample at least 5 clients
    log(INFO, "Waiting for at least 5 clients to connect...")
    while True:
        nids = driver.get_node_ids()
        if len(nids) >= 5:
            break
        time.sleep(0.1)

    # Create the SecAgg+ aggregator
    sa_aggregator = SecAggPlusAggregator(
        driver=driver,
        context=context,
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        timeout=context.run_config["timeout"],
        clipping_range=8.0,
        on_send=on_send,
        on_receive=on_receive,
        on_stage_complete=on_stage_complete,
    )
    msgs = [
        driver.create_message(
            content=RecordSet(),  # Empty message
            message_type=MessageType.QUERY,
            dst_node_id=nid,
            group_id="",
        )
        for nid in nids
    ]
    msgs[0].metadata.group_id = "drop"
    msg = sa_aggregator.aggregate(msgs)

    arr = msg.content.parameters_records["simple_pr"]["simple_array"]
    log(INFO, f"Received aggregated array: {arr.numpy()}")


# Example `on_send`/`on_receive`/`on_stage_complete` callback functions
def on_send(msgs: Iterable[Message], state: SecAggPlusAggregatorState) -> None:
    """Intercept messages before sending."""
    log(INFO, "Intercepted messages before sending.")


def on_receive(msgs: Iterable[Message], state: SecAggPlusAggregatorState) -> None:
    """Intercept reply messages after receiving."""
    log(INFO, "Intercepted reply messages after receiving.")


def on_stage_complete(success: bool, state: SecAggPlusAggregatorState) -> None:
    """Handle stage completion event."""
    log(INFO, "Handled stage completion event.")
