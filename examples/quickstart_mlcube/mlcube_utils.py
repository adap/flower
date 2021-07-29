import os
import sys
import subprocess
import tensorflow as tf
import json

MODULE_PATH = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(MODULE_PATH)
MLCUBE_DIR = os.path.join(MODULE_DIR, "mlcube")
MLCUBE_WORKSPACE_DIR = os.path.join(MLCUBE_DIR, "workspace")


def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def workspace_path(path: str, is_file=True) -> str:
    """Return filepath and create directories if required."""
    full_path = os.path.join(MLCUBE_WORKSPACE_DIR, path)
    dir_path = os.path.dirname(full_path) if is_file else full_path
    create_directory(dir_path)
    return full_path


model_in_filepath = workspace_path("model_in/mnist_model")
model_out_filepath = workspace_path("model_out/mnist_model")
train_metrics_filepath = workspace_path("metrics/train_metrics.json")
evaluate_metrics_filepath = workspace_path("metrics/evaluate_metrics.json")


def run_task(task_name: str):
    """Run mlcube task and return if successful."""
    command = [
        "mlcube_docker",
        "run",
        "--mlcube=.",
        "--platform=platforms/docker.yaml",
        f"--task=run/{task_name}.yaml",
    ]

    print(" ".join(command))
    process = subprocess.Popen(
        command, cwd=MLCUBE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.buffer.write(c)

    process.communicate()

    if process.returncode != 0:
        raise Exception("MLCube task %s failed" % task_name)


def save_parameteras_as_model(parameters):
    """Write model to $WORKSPACE/model_in/mnist_model from parameters."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.set_weights(parameters)
    model.save(model_in_filepath)



def load_model_parameters():
    """Load and return model parameters"""
    model = tf.keras.models.load_model(model_in_filepath)
    return model.get_weights()


def load_train_metrics():
    """Load and return metrics."""
    with open(train_metrics_filepath, "r") as f:
        data = json.load(f)

    data["loss"] = float(data["loss"])
    data["accuracy"] = float(data["accuracy"])
    data["num_examples"] = int(data["num_examples"])

    return data


def load_evaluate_metrics():
    """Load and return metrics."""
    with open(evaluate_metrics_filepath, "r") as f:
        data = json.load(f)

    data["loss"] = float(data["loss"])
    data["accuracy"] = float(data["accuracy"])
    data["num_examples"] = int(data["num_examples"])


    return data
