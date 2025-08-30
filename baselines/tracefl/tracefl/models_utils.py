"""Utility functions for model management in federated learning.

This module provides utility functions for model initialization, parameter management,
and evaluation in a federated learning setting. It supports both transformer-based
models (like BERT) and CNN models (like ResNet), with functionality for:
- Model initialization and configuration
- Parameter extraction and setting
- Model evaluation and metrics computation
- Dataset preparation and batch processing
"""

import logging

import evaluate
import numpy as np
import torch
import torchvision
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _compute_metrics(eval_pred):
    """Compute evaluation metrics from predictions and labels."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    correct_predictions = predictions == labels
    incorrect_predictions = ~correct_predictions

    correct_indices = np.where(correct_predictions)[0]
    correct_indices = torch.from_numpy(correct_indices)

    incorrect_indices = np.where(incorrect_predictions)[0]
    incorrect_indices = torch.from_numpy(incorrect_indices)

    return {
        "accuracy": metric.compute(predictions=predictions, references=labels),
        "correct_indices": correct_indices,
        "actual_labels": labels,
        "incorrect_indices": incorrect_indices,
        "predicted_labels": predictions,
    }


def _get_inputs_labels_from_batch(batch):
    """Extract input data and labels from a batch."""
    if "pixel_values" in batch:
        return batch["pixel_values"], batch["label"]
    x, y = batch
    return x, y


def initialize_model(name, cfg_dataset):
    """Initialize and configure the model based on its name and dataset configuration.

    Args:
        name: Name of the model to initialize
        cfg_dataset: Dataset configuration containing model parameters

    Returns
    -------
        Dictionary containing the initialized model and number of classes
    """
    model_dict = {"model": None, "num_classes": cfg_dataset.num_classes}
    if name in [
        "squeezebert/squeezebert-uncased",
        "openai-community/openai-gpt",
        "Intel/dynamic_tinybert",
        "google-bert/bert-base-cased",
        "microsoft/MiniLM-L12-H384-uncased",
        "distilbert/distilbert-base-uncased",
    ]:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=name,
            num_labels=cfg_dataset.num_classes,
        )

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id

        model_dict["model"] = model.cpu()

    elif name.find("resnet") != -1:
        if "resnet18" == name:
            model = torchvision.models.resnet18(weights=None)
        elif "resnet34" == name:
            model = torchvision.models.resnet34(weights=None)
        elif "resnet50" == name:
            model = torchvision.models.resnet50(weights=None)
        elif "resnet101" == name:
            model = torchvision.models.resnet101(weights=None)
        elif "resnet152" == name:
            model = torchvision.models.resnet152(weights=None)
        else:
            raise ValueError(f"Failed to initialize model {name}")

        if cfg_dataset.channels == 1:
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()

    elif name == "densenet121":
        model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        if cfg_dataset.channels == 1:
            logging.info(
                "Changing the first layer of densenet model to accept 1 channel"
            )
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()
    else:
        raise ValueError(f"Model {name} not supported")

    return model_dict


def get_parameters(model):
    """Extract and return the parameters of a PyTorch model as a list of NumPy arrays.

    Args:
        model: PyTorch model to extract parameters from

    Returns
    -------
        List of NumPy arrays containing model parameters
    """
    model = model.cpu()
    return [val.cpu().detach().clone().numpy() for _, val in model.state_dict().items()]


def set_parameters(net, parameters):
    """Set the parameters of a PyTorch model from a list of NumPy arrays."""
    net = net.cpu()
    params_dict = zip(net.state_dict().keys(), parameters)
    new_state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    net.load_state_dict(new_state_dict, strict=False)
