from typing import List
import time

import flwr as fl
from flwr.common import (
    Context,
    NDArrays,
    Message,
    MessageType,
    Metrics,
    RecordSet,
    ConfigsRecord,
    DEFAULT_TTL,
)
from flwr.server import Driver

from task import Net
from low_level_utils import (
    parameters_to_pytorch_state_dict,
    pytorch_to_parameter_record,
)


# Run via `flower-server-app server:app`
app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    """This is a stub example that simply sends and receives messages."""
    print("Starting test run")
    global_model = Net()
    for server_round in range(3):
        print(f"Commencing server round {server_round + 1}")

        # Get node IDs
        node_ids = driver.get_node_ids()

        # Create messages
        recordset = RecordSet()

        # add model parameters to record
        recordset.parameters_records["fancy_model"] = pytorch_to_parameter_record(
            global_model
        )
        # Add a config
        recordset.configs_records["fancy_config"] = ConfigsRecord({"epochs": 1})

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

        # Ignore messages with Error
        all_replies = [
            msg
            for msg in all_replies
            if msg.has_content()
        ]
        print(f"Received {len(all_replies)} results")

        # print metrics
        for reply in all_replies:
            print(reply.content.metrics_records)

        # Do some primitive aggregation directly with PyTorch state_dicts
        new_global_model = Net()
        new_state_dict = global_model.state_dict()

        received_state_dicts = [
            parameters_to_pytorch_state_dict(
                reply.content.parameters_records["fancy_model_returned"]
            )
            for reply in all_replies
        ]

        for sd_key in new_state_dict.keys():
            new_state_dict[sd_key] = sum(r_s_d[sd_key] for r_s_d in received_state_dicts)/ len(received_state_dicts)
    
        # apply aggregated state_dicts
        new_global_model.load_state_dict(new_state_dict)

        #? Or do somethig else

        global_model = new_global_model

