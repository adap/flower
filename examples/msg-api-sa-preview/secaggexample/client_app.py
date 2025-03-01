"""secaggexample: A Flower with SecAgg+ app."""

import numpy as np

from flwr.client import ClientApp
from flwr.common import Array, Context, Message, ParametersRecord, RecordSet

from .secaggplus_base_mod import secaggplus_base_mod

# Flower ClientApp
app = ClientApp()


@app.query(mods=[secaggplus_base_mod])
def simple_query(msg: Message, ctxt: Context) -> Message:
    """Simple query function."""
    pr = ParametersRecord()
    pr["simple_array"] = Array(np.array([0.5, 1, 2, 4]))
    print(f"Sending simple array: {pr['simple_array'].numpy()}")
    content = RecordSet()
    content["simple_pr"] = pr
    return msg.create_reply(content=content)
