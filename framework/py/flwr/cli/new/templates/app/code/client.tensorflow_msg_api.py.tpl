"""$project_name: A Flower / $framework_str app."""

from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

from $import_name.task import load_data, load_model

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = load_model()
    global_ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(global_ndarrays)

    # Read from config
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, _, _ = load_data(partition_id, num_partitions)

    # Fit the model to the data
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.get_weights())
    metrics = {
        "num-examples": len(x_train),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = load_model()
    global_ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(global_ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, x_test, y_test = load_data(partition_id, num_partitions)

    # Call the evaluation function
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Construct and return reply Message
    metrics = {
        "eval_loss": loss,
        "eval_acc": accuracy,
        "num-examples": len(x_test),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
