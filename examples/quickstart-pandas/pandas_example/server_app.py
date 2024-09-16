"""pandas_example: A Flower / Pandas app."""

import random
import time
from logging import INFO

import numpy as np

from flwr.common import Context, MessageType, RecordSet
from flwr.common.logger import log
from flwr.server import Driver, ServerApp

app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

    num_rounds = context.run_config["num-server-rounds"]
    num_client_nodes_per_round = 2

    for server_round in range(num_rounds):
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Loop and wait until enough nodes are available.
        max_connection_attempts = 10
        attempt = 1
        while attempt < max_connection_attempts:
            all_node_ids = driver.get_node_ids()
            log(
                INFO,
                "(Attempt %s of %s) Connected to %s client nodes: %s",
                attempt,
                max_connection_attempts,
                len(all_node_ids),
                all_node_ids,
            )
            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                node_ids = random.sample(all_node_ids, num_client_nodes_per_round)
                break
            attempt += 1
            time.sleep(3)

        # Create messages
        recordset = RecordSet()
        messages = []
        for node_id in node_ids:  # one message for each node
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        # Send messages and wait for all results
        replies = driver.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Post process results from queries
        aggregated_metrics = {}
        for rep in replies:
            query_results = rep.content.metrics_records["query_results"]
            # Sum metrics
            for k, v in query_results.items():
                if k in ["SepalLengthCm", "SepalWidthCm"]:
                    if k in aggregated_metrics:
                        aggregated_metrics[k] += np.array(v)
                    else:
                        aggregated_metrics[k] = np.array(v)


        # Display aggregated metrics
        log(INFO, "Aggregated metrics: %s", aggregated_metrics)
