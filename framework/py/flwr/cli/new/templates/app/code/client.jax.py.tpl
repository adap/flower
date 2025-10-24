"""$project_name: A Flower / $framework_str app."""

import jax
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from $import_name.task import evaluation as evaluation_fn
from $import_name.task import get_params, load_data, load_model, loss_fn, set_params
from $import_name.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Read from config
    input_dim = context.run_config["input-dim"]

    # Load data and model
    train_x, train_y, _, _ = load_data()
    model = load_model((input_dim,))
    grad_fn = jax.grad(loss_fn)

    # Set model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_params(model, ndarrays)

    # Train the model on local data
    model, loss, num_examples = train_fn(model, grad_fn, train_x, train_y)

    # Construct and return reply Message
    model_record = ArrayRecord(get_params(model))
    metrics = {
        "train_loss": float(loss),
        "num-examples": num_examples,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Read from config
    input_dim = context.run_config["input-dim"]

    # Load data and model
    _, _, test_x, test_y = load_data()
    model = load_model((input_dim,))
    grad_fn = jax.grad(loss_fn)

    # Set model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_params(model, ndarrays)

    # Evaluate the model on local data
    loss, num_examples = evaluation_fn(model, grad_fn, test_x, test_y)

    # Construct and return reply Message
    metrics = {
        "test_loss": float(loss),
        "num-examples": num_examples,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
