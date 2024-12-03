"""secaggexample: A Flower with SecAgg+ app."""

import numpy as np

from flwr.client import ClientApp
from flwr.client.mod import secaggplus_base_mod
from flwr.common import Context, Message, ParametersRecord, RecordSet, array_from_numpy

# Flower ClientApp
app = ClientApp(mods=[secaggplus_base_mod])


@app.query()
def simple_query(msg: Message, ctxt: Context) -> Message:
    """Simple query function."""
    if msg.metadata.group_id == "drop":
        print("Dropping out")
        return msg.create_error_reply("Dropping out")

    pr = ParametersRecord()
    pr["simple_array"] = array_from_numpy(np.array([0.5, 1, 2]))
    print(f"Sending simple array: {pr['simple_array'].numpy()}")
    content = RecordSet()
    content.parameters_records["simple_pr"] = pr
    return msg.create_reply(content=content)
