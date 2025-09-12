"""Model training and evaluation functionality for TraceFL.

This module contains functions for training and evaluating neural network models in the
TraceFL federated learning system. It supports both CNN and transformer models.
"""

import logging

from transformers import Trainer, TrainingArguments

from tracefl.models_cnn_trainer import _test_cnn, _test_cnn_hf_trainer, _train_cnn
from tracefl.models_utils import _compute_metrics


def _train_transformer(model, train_data, _test_data, device, cfg):
    """Train a transformer model using HuggingFace Trainer.

    Args:
        model: The transformer model to train
        train_data: Training dataset
        _test_data: Test dataset (unused, kept for API compatibility)
        device: Device to train on
        cfg: Configuration object

    Returns
    -------
        Dictionary containing model and training metrics
    """
    model_dict = {"model": model, "num_classes": cfg.num_classes}
    net = model.to(device)

    training_args = TrainingArguments(
        output_dir="training_output",
        lr_scheduler_type="constant",
        num_train_epochs=cfg.epochs,
        eval_strategy="no",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        disable_tqdm=True,
        report_to="none",
    )

    trainer = Trainer(
        model=net,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=train_data,
        compute_metrics=_compute_metrics,
    )

    if "audio_feature_extractor" in model_dict:
        trainer.tokenizer = model_dict["audio_feature_extractor"]

    trainer.train()
    r = trainer.evaluate(eval_dataset=train_data)
    net = net.cpu()

    eval_accuracy = r.get("eval_accuracy", {})
    if isinstance(eval_accuracy, dict):
        accuracy = eval_accuracy.get("accuracy", 0.0)
    else:
        accuracy = eval_accuracy

    return {
        "model": net,
        "train_metrics": {
            "train_loss": r.get("eval_loss", 0.0),
            "train_accuracy": accuracy,
        },
        "test_metrics": {
            "eval_loss": r.get("eval_loss", 0.0),
            "eval_accuracy": accuracy,
        },
    }


def _test_transformer_model(args):
    """Evaluate a transformer model."""
    logging.debug("Evaluating transformer model")
    model_dict, central_server_test_data, batch_size = (
        args["model_dict"],
        args["test_data"],
        args["batch_size"],
    )
    net = model_dict["model"]
    testing_args = TrainingArguments(
        logging_strategy="steps",
        output_dir=".",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=batch_size,
        disable_tqdm=True,
        report_to="none",
    )

    tester = Trainer(
        model=net,
        args=testing_args,
        compute_metrics=_compute_metrics,
        eval_dataset=central_server_test_data,
    )

    if "audio_feature_extractor" in model_dict:
        tester.tokenizer = model_dict["audio_feature_extractor"]

    logging.debug("length of eval dataset: %s", len(central_server_test_data))
    r = tester.evaluate()
    net = net.cpu()
    return r


def global_model_eval(model_arch, model_dict):
    """Evaluate the global model.

    Args:
        model_arch: Model architecture name
        model_dict: Dictionary containing model and metadata

    Returns
    -------
        Dictionary of evaluation metrics
    """
    if model_arch == "cnn":
        d = _test_cnn(
            model_dict["model"], test_data=model_dict["test_data"], device="cpu"
        )
    elif model_arch == "transformer":
        d = _test_transformer_model(
            {
                "model_dict": model_dict,
                "test_data": model_dict["test_data"],
                "batch_size": model_dict["batch_size"],
            }
        )
    else:
        raise ValueError(f"Unsupported architecture: {model_arch}")

    return {
        "loss": d["eval_loss"],
        "accuracy": d["eval_accuracy"],
    }


def test_neural_network(arch, global_net_dict, server_testdata, batch_size=32):
    """Evaluate a trained model using the appropriate evaluation strategy."""
    if arch == "cnn":
        d = _test_cnn_hf_trainer(
            global_net_dict,
            central_server_test_data=server_testdata,
            batch_size=batch_size,
        )
    elif arch == "transformer":
        d = _test_transformer_model(
            {
                "model_dict": global_net_dict,
                "test_data": server_testdata,
                "batch_size": batch_size,
            }
        )
    else:
        raise ValueError(f"Architecture {arch} not supported")

    eval_accuracy = d.get("eval_accuracy", {})
    if isinstance(eval_accuracy, dict):
        accuracy = eval_accuracy.get("accuracy", 0.0)
    else:
        accuracy = eval_accuracy

    d["loss"] = d.get("eval_loss", 0.0)
    d["accuracy"] = accuracy
    return d


def train_neural_network(tconfig):
    """Train either a CNN or Transformer model depending on the config."""
    if tconfig["arch"] == "cnn":
        return _train_cnn(tconfig)
    if tconfig["arch"] == "transformer":
        return _train_transformer(
            tconfig["model"],
            tconfig["train_data"],
            tconfig["test_data"],
            tconfig["device"],
            tconfig,
        )
    raise ValueError(f"Architecture {tconfig['arch']} not supported")
