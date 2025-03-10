"""secaggexample: A Flower with SecAgg+ app."""

import time
from collections.abc import Iterable
from logging import DEBUG, INFO

from flwr.common import Context, Message, RecordSet, log
from flwr.common.constant import MessageType
from flwr.common.logger import update_console_handler
from flwr.server import Driver, ServerApp

from .secaggplus_aggregator import SecAggPlusAggregatorState, SecAggPlusInsAggregator

# Flower ServerApp
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Create the SecAgg+ aggregator
    sa_aggregator = SecAggPlusInsAggregator(
        driver=driver,
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        timeout=context.run_config["timeout"],
        clipping_range=8.0,
        on_send=on_send,
        on_receive=on_receive,
        on_stage_complete=on_stage_complete,
    )

    # Show debug logs
    update_console_handler(DEBUG, True, True)

    # Sample at least 5 clients
    min_clients: int = context.run_config["min-clients"]
    log(INFO, "Waiting for at least %d clients to connect...", min_clients)
    while True:
        nids = driver.get_node_ids()
        if len(nids) >= min_clients:
            break
        time.sleep(0.1)

    # Send query messages and aggregate reply messages
    rs = sa_aggregator.aggregate(
        [
            driver.create_message(
                content=RecordSet(),  # Empty message
                message_type=MessageType.QUERY,
                dst_node_id=nid,
                group_id="",
            )
            for nid in nids
        ]
    )

    arr = rs["simple_pr"]["simple_array"]
    log(INFO, f"Received aggregated array: {arr.numpy()}")


# Example `on_send`/`on_receive`/`on_stage_complete` callback functions
def on_send(msgs: Iterable[Message], state: SecAggPlusAggregatorState) -> None:
    """Intercept messages before sending."""
    log(INFO, "Intercepted messages before sending.")


def on_receive(msgs: Iterable[Message], state: SecAggPlusAggregatorState) -> None:
    """Intercept reply messages after receiving."""
    log(INFO, "Intercepted reply messages after receiving.")
    for msg in msgs:
        if msg.content.parameters_records:
            arr = msg.content["simple_pr"]["simple_array"]
            print("Received array:", arr.numpy())


def on_stage_complete(success: bool, state: SecAggPlusAggregatorState) -> None:
    """Handle stage completion event."""
    log(INFO, "Handled stage completion event.")
