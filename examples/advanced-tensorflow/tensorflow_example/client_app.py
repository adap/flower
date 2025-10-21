"""tensorflow-example: A Flower / Tensorflow app."""

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Array, ArrayRecord, Context, RecordDict

from tensorflow_example.task import load_data, load_model

# Flower ClientApp
app = ClientApp()
classification_head_name = "classification-head"

from tensorflow_example.task import load_data, load_model


def save_layer_weights_to_state(state: RecordDict, model):
    """Save last layer weights to state."""
    state_dict_arrays = {}
    # Get weights from the last layer
    list_weights = model.get_layer("dense").get_weights()

    # Add to RecordDict (replace if already exists)
    state[classification_head_name] = ArrayRecord(list_weights)


def load_layer_weights_from_state(state: RecordDict, model):
    """Load last layer weights to state."""
    if classification_head_name not in state:
        return

    # Apply weights
    list_weights = state[classification_head_name].to_numpy_ndarrays()
    model.get_layer("dense").set_weights(list_weights)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    keras.backend.clear_session()
    # Read config
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]

    # Load model and apply received weights
    lr = msg.content["config"]["lr"]
    model = load_model(learning_rate=lr)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    # Restore this client's previously saved classification layer weights
    # (no action if this is the first round it participates in)
    load_layer_weights_from_state(context.state, model)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, _, _ = load_data(partition_id, num_partitions)

    # Call the training function
    history = model.fit(
        x_train,
        y_train,
        epochs=local_epochs,
        batch_size=batch_size,
        verbose=0,
    )
    # Get final training loss and accuracy
    train_loss = history.history["loss"][-1] if "loss" in history.history else None
    train_acc = history.history.get("accuracy")
    train_acc = train_acc[-1] if train_acc is not None else None

    # Save classification head in `context.state` to use in future rounds
    save_layer_weights_to_state(context.state, model)

    # Construct and return reply Message
    model_record = ArrayRecord(model.get_weights())
    metrics = {"num-examples": len(x_train)}
    if train_loss is not None:
        metrics["train_loss"] = train_loss
    if train_acc is not None:
        metrics["train_acc"] = train_acc
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    keras.backend.clear_session()

    # Load model and apply received weights
    model = load_model()
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    # Restore this client's previously saved classification layer weights
    # (no action if this is the first round it participates in)
    load_layer_weights_from_state(context.state, model)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, x_test, y_test = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(x_test),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
