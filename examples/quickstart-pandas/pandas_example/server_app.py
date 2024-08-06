"""pandas_example: A Flower / Pandas app."""

from logging import INFO
import numpy as np
import flwr as fl
from flwr.common import (
    Context,
    MessageType,
    RecordSet,
)
from flwr.common.logger import log
from flwr.server import Driver


app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

    num_rounds = context.run_config["num-server-rounds"]
    for server_round in range(num_rounds):
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Get IDs of nodes available
        node_ids = driver.get_node_ids()

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
        aggregated_histograms = {}
        for rep in replies:
            query_results = rep.content.metrics_records["query_results"]
            for k, v in query_results.items():
                if k in aggregated_histograms:
                    aggregated_histograms[k] += np.array(v)
                else:
                    aggregated_histograms[k] = np.array(v)

        # Aggregate partial histograms
        log(INFO, "Aggregated histograms: %s", aggregated_histograms)
