from typing import List, Tuple, Dict
import random
import time

import flwr as fl
from flwr.common import (
    Context,
    NDArrays,
    Message,
    MessageType,
    Metrics,
    RecordSet,
    DEFAULT_TTL,
)
from flwr.server import Driver


# Run via `flower-server-app server:app`
app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    """This is a stub example that simply sends and receives messages."""
    print("Starting test run")
    for server_round in range(3):
        print(f"Commencing server round {server_round + 1}")

        # Get node IDs
        node_ids = driver.get_node_ids()

        # Create messages
        recordset = RecordSet()
        messages = []
        for node_id in node_ids:
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
                group_id=str(server_round),
                ttl=DEFAULT_TTL,
            )
            messages.append(message)

        # Send messages
        message_ids = driver.push_messages(messages)
        print(f"Pushed {len(message_ids)} messages: {message_ids}")

        # Wait for results, ignore empty message_ids
        message_ids = [message_id for message_id in message_ids if message_id != ""]
        all_replies: List[Message] = []
        while True:
            replies = driver.pull_messages(message_ids=message_ids)
            print(f"Got {len(replies)} results")
            all_replies += replies
            if len(all_replies) == len(message_ids):
                break
            time.sleep(3)

        print(f"Received {len(all_replies)} results")
