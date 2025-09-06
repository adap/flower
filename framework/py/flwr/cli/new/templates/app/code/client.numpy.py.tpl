"""$project_name: A Flower / $framework_str app."""

import numpy as np
from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # The model is the global arrays
    model = msg.content["arrays"].to_numpy_ndarrays()

    # Apply a random transformation to the model
    model = [m + np.random.rand(*m.shape) for m in model]

    # Construct and return reply Message
    model_record = ArrayRecord(model)
    metrics = {
        "random_metric": np.random.rand(),
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # The model is the global arrays
    model = msg.content["arrays"].to_numpy_ndarrays()

    # Return reply Message
    metrics = {
        "random_metric": np.random.rand(3).tolist(),
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
