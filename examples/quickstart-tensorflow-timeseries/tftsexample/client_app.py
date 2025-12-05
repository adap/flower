"""timeseries: A Flower / TensorFlow app."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from timeseries.task import load_data, load_model

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = load_model()
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(ndarrays)

    # Read from config
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    tf_train, _ = load_data(partition_id, num_partitions, batch_size)

    # Train the model on local data
    history = model.fit(
        tf_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Get final training loss and accuracy
    train_loss = history.history["loss"][-1] if "loss" in history.history else None
    train_acc = history.history.get("accuracy")
    train_acc = train_acc[-1] if train_acc is not None else None

    # Construct and return reply Message
    model_record = ArrayRecord(model.get_weights())
    metrics = {"num-examples": int(32000*0.8)}
    if train_loss is not None:
        metrics["train_loss"] = train_loss
    if train_acc is not None:
        metrics["train_acc"] = train_acc
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = load_model()
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    num_partitions = context.node_config["num-partitions"]
    _, tf_test = load_data(partition_id, num_partitions, batch_size)

    # Evaluate the model on local data
    loss, accuracy = model.evaluate(tf_test, verbose=0)

    # Construct and return reply Message
    metrics = {
        "eval_loss": loss,
        "eval_acc": accuracy,
        "num-examples": int(32000*0.2),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
