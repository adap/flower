"""$project_name: A Flower / $framework_str app."""

import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from $import_name.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, _, y_train, _ = load_data(partition_id, num_partitions)

    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Train the model on local data
        model.fit(X_train, y_train)

    # Let's compute train loss
    y_train_pred_proba = model.predict_proba(X_train)
    train_logloss = log_loss(y_train, y_train_pred_proba)

    # Construct and return reply Message
    ndarrays = get_model_params(model)
    model_record = ArrayRecord(ndarrays)
    metrics = {"num-examples": len(X_train), "train_logloss": train_logloss}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on test data."""

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, X_test, _, y_test = load_data(partition_id, num_partitions)

    # Evaluate the model on local data
    y_train_pred = model.predict(X_test)
    y_train_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_train_pred)
    loss = log_loss(y_test, y_train_pred_proba)
    precision = precision_score(y_test, y_train_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_train_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_train_pred, average="macro", zero_division=0)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_test),
        "test_logloss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
