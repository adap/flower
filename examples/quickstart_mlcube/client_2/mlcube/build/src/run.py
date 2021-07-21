"""Example entry point script compatible with MLCube."""
import argparse
import os
from typing import Any, Optional, Dict, Tuple
import logging
from enum import Enum
from typing import List
import tensorflow as tf
import pickle
from logger import configure_logger
import uuid


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Task(str, Enum):
    GET_PARAMETERS = "get_parameters"
    FIT = "fit"
    EVALUATE = "evaluate"


def create_directory(path: str, file: bool) -> None:
    _path = os.path.dirname(path) if file else path
    os.makedirs(_path, exist_ok=True)


def setup(
    task_args: List[str], task_name
) -> Tuple[Dict[str, Optional[str]], logging.Logger]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir", "--log-dir", type=str, default=None, help="Log directory."
    )
    parser.add_argument(
        "--input_file", "--input-file", type=str, default=None, help="Inputs file.",
    )
    parser.add_argument(
        "--output_file", "--output-file", type=str, default=None, help="Outputs file.",
    )
    args: Dict[str, Optional[str]] = parser.parse_args(args=task_args)

    create_directory(path=args.log_dir, file=False)
    logger = configure_logger(args.log_dir, task_name)

    return args, logger


def create_model():
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def mlcube_task_get_parameters(task_args: List[str]) -> None:
    """ Task: get_parameters."""
    args, logger = setup(task_args, Task.GET_PARAMETERS)
    logger.info(f"Starting '{Task.GET_PARAMETERS}' task")

    model = create_model()
    result = {"parameters": model.get_weights()}
    pickle.dump(result, open(args.output_file, "wb"))


def mlcube_task_fit(task_args: List[str]) -> None:
    """ Task: fit."""
    args, logger = setup(task_args, Task.FIT)
    logger.info(f"Starting '{Task.FIT}' task")

    inputs = pickle.load(open(args.input_file, "rb"))
    model = create_model()
    model.set_weights(inputs["parameters"])

    # Load CIFAR-10 dataset
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=1)

    result = {
        "parameters": model.get_weights(),
        "num_examples": len(x_train),
        "config": {},
    }

    pickle.dump(result, open(args.output_file, "wb"))


def mlcube_task_evaluate(task_args: List[str]) -> None:
    """ Task: evaluate."""
    args, logger = setup(task_args, Task.EVALUATE)
    logger.info(f"Starting '{Task.EVALUATE}' task")

    inputs = pickle.load(open(args.input_file, "rb"))
    model = create_model()
    model.set_weights(inputs["parameters"])

    # Load CIFAR-10 testset
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    loss, accuracy = model.evaluate(x_test, y_test)

    result = {
        "loss": loss,
        "num_examples": len(x_test),
        "config": {"accuracy": accuracy},
    }

    pickle.dump(result, open(args.output_file, "wb"))


def main():
    # Every MLCuber runner passes a task name as the first argument. Other arguments are task-specific.
    parser = argparse.ArgumentParser()
    parser.add_argument("mlcube_task", type=str, help="Task for this MLCube.")

    # The `mlcube_args` contains task name (mlcube_args.mlcube_task)
    # The `task_args` list needs to be parsed later when task name is known
    mlcube_args, task_args = parser.parse_known_args()

    if mlcube_args.mlcube_task == Task.GET_PARAMETERS:
        mlcube_task_get_parameters(task_args)
    elif mlcube_args.mlcube_task == Task.FIT:
        mlcube_task_fit(task_args)
    elif mlcube_args.mlcube_task == Task.EVALUATE:
        mlcube_task_evaluate(task_args)
    else:
        raise ValueError(f"Unknown task: '{mlcube_args.mlcube_task}'")


if __name__ == "__main__":
    main()
