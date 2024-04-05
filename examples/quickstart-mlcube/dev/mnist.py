"""
https://www.tensorflow.org/tutorials/quickstart/beginner
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import logging
import logging.config
import os
import sys
from enum import Enum
from typing import List

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.utils import get_file

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Remove warning messages

logger = logging.getLogger(__name__)


class Task(str, Enum):
    DownloadData = "download"
    Train = "train"
    Evaluate = "evaluate"


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download(task_args: List[str]) -> None:
    """Task: download.

    Input parameters:
        --data_dir
    """
    logger.info(f"Starting '{Task.DownloadData}' task")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        type=str,
        default=None,
        help="Path to a dataset file.",
    )
    args = parser.parse_args(args=task_args)

    if args.data_dir is None:
        raise ValueError(
            "Data directory is not specified (did you use --data-dir=PATH?)"
        )
    os.makedirs(args.data_dir, exist_ok=True)

    if not args.data_dir.startswith("/"):
        logger.warning("Data directory seems to be a relative path.")

    data_file = os.path.join(args.data_dir, "mnist.npz")
    if os.path.exists(data_file):
        logger.info("MNIST data has already been download (file exists: %s)", data_file)
        return

    data_file = get_file(
        fname=data_file,
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        file_hash="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
    )

    if not os.path.isfile(data_file):
        raise ValueError(
            f"MNIST dataset has not been downloaded - dataset file does not exist: {data_file}"
        )
    else:
        logger.info("MNIST dataset has been downloaded.")
    logger.info("The '%s' task has been completed.", Task.DownloadData)


def train(task_args: List[str]) -> None:
    """Task: train.

    Input parameters:
        --data_dir, --log_dir, --model_dir, --parameters_file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "--data-dir", type=str, default=None, help="Dataset path."
    )
    parser.add_argument(
        "--model_in",
        "--model-in",
        type=str,
        default=None,
        help="Model output directory.",
    )
    parser.add_argument(
        "--model_dir",
        "--model-dir",
        type=str,
        default=None,
        help="Model output directory.",
    )
    parser.add_argument(
        "--parameters_file",
        "--parameters-file",
        type=str,
        default=None,
        help="Parameters default values.",
    )
    parser.add_argument(
        "--metrics",
        "--metrics",
        type=str,
        default=None,
        help="Parameters default values.",
    )

    args = parser.parse_args(args=task_args)

    with open(args.parameters_file, "r") as stream:
        parameters = yaml.load(stream, Loader=yaml.FullLoader)
    logger.info("Parameters have been read (%s).", args.parameters_file)

    dataset_file = os.path.join(args.data_dir, "mnist.npz")
    with np.load(dataset_file, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
    x_train = x_train / 255.0
    logger.info("Dataset has been loaded (%s).", dataset_file)

    if args.model_in != "" and len(os.listdir(args.model_in)) != 0:
        # Load from checkpoint;
        model = tf.keras.models.load_model(os.path.join(args.model_in, "mnist_model"))
    else:
        # if no model given on CLI, create a new one
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

    logger.info("Model has been built.")

    model.compile(
        optimizer=parameters.get("optimizer", "adam"),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Model has been compiled.")

    # Train and evaluate
    history = model.fit(
        x_train,
        y_train,
        batch_size=parameters.get("batch_size", 32),
        epochs=parameters.get("epochs", 5),
    )
    logger.info("Model has been trained.")

    with open(args.metrics, "w") as f:
        data_json = {
            "loss": str(history.history["loss"][-1]),
            "accuracy": str(history.history["accuracy"][-1]),
        }
        json.dump(data_json, f)

    os.makedirs(args.model_dir, exist_ok=True)
    model.save(os.path.join(args.model_dir, "mnist_model"))
    logger.info("Model has been saved.")


def evaluate(task_args: List[str]) -> None:
    """Task: train.

    Input parameters:
        --data_dir, --log_dir, --model_dir, --parameters_file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "--data-dir", type=str, default=None, help="Dataset path."
    )
    parser.add_argument(
        "--model_in",
        "--model-in",
        type=str,
        default=None,
        help="Model output directory.",
    )
    parser.add_argument(
        "--parameters_file",
        "--parameters-file",
        type=str,
        default=None,
        help="Parameters default values.",
    )
    parser.add_argument(
        "--metrics",
        "--metrics",
        type=str,
        default=None,
        help="Parameters default values.",
    )
    args = parser.parse_args(args=task_args)

    with open(args.parameters_file, "r") as stream:
        parameters = yaml.load(stream, Loader=yaml.FullLoader)
        logger.info("Parameters have been read (%s).", parameters)

    dataset_file = os.path.join(args.data_dir, "mnist.npz")
    with np.load(dataset_file, allow_pickle=True) as f:
        x_test, y_test = f["x_test"], f["y_test"]
    x_test = x_test / 255.0
    logger.info("Dataset has been loaded (%s).", dataset_file)

    model = tf.keras.models.load_model(os.path.join(args.model_in, "mnist_model"))

    eval_result = model.evaluate(x_test, y_test)

    with open(args.metrics, "w") as f:
        data_json = {"loss": str(eval_result[0]), "accuracy": str(eval_result[1])}
        json.dump(data_json, f)

    logger.info("Model has been evaluated.")


def main():
    """mnist.py task task_specific_parameters..."""
    # noinspection PyBroadException
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("mlcube_task", type=str, help="Task for this MLCube.")
        parser.add_argument(
            "--log_dir", "--log-dir", type=str, required=True, help="Logging directory."
        )
        mlcube_args, task_args = parser.parse_known_args()

        os.makedirs(mlcube_args.log_dir, exist_ok=True)
        logger_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "file_handler": {
                    "class": "logging.FileHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "filename": os.path.join(
                        mlcube_args.log_dir,
                        f"mlcube_mnist_{mlcube_args.mlcube_task}.log",
                    ),
                }
            },
            "loggers": {
                "": {"level": "INFO", "handlers": ["file_handler"]},
                "__main__": {"level": "NOTSET", "propagate": "yes"},
                "tensorflow": {"level": "NOTSET", "propagate": "yes"},
            },
        }
        logging.config.dictConfig(logger_config)

        if mlcube_args.mlcube_task == Task.DownloadData:
            download(task_args)
        elif mlcube_args.mlcube_task == Task.Train:
            train(task_args)
        elif mlcube_args.mlcube_task == Task.Evaluate:
            evaluate(task_args)
        else:
            raise ValueError(f"Unknown task: {task_args}")
    except Exception as err:
        logger.exception(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
