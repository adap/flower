"""CNN model trainer module for federated learning.

This module provides training and evaluation functionality for CNN models in a federated
learning setting. It includes both a custom CNNTrainer class that extends HuggingFace's
Trainer, as well as utility functions for training and evaluating CNN models using
PyTorch's native training loops.
"""

import gc
import logging

import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, Trainer, TrainingArguments

from tracefl.models_utils import _compute_metrics, _get_inputs_labels_from_batch


class CNNTrainer(Trainer):
    """CNN Model Trainer for federated learning tasks."""

    def __init__(
        self,
        model,
        args,
        *,
        train_data=None,
        test_data=None,
        compute_metrics=None,
        data_collator=None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=test_data,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for the model on the given inputs.

        Args:
            model: The model to compute loss for
            inputs: Dictionary containing 'labels' and 'pixel_values'
            return_outputs: Whether to return model outputs along with loss

        Returns
        -------
            Tuple of (loss, outputs) if return_outputs is True, else just loss
        """
        labels = inputs.get("labels")
        batch_inputs = inputs.get("pixel_values")
        outputs = model(batch_inputs)
        logits = outputs
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Perform a prediction step on the model.

        Args:
            model: The model to use for prediction
            inputs: Input data for prediction
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Optional list of keys to ignore in inputs
                (unused, kept for API compatibility)

        Returns
        -------
            Tuple of (loss, labels, logits) if prediction_loss_only is False,
            else just (loss, None, None)
        """
        loss = None
        labels = None
        logits = None
        with torch.no_grad():
            logits = model(inputs["pixel_values"])
            if "labels" in inputs:
                labels = inputs.get("labels")
                loss = torch.nn.CrossEntropyLoss()(logits, labels)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def _train_cnn(tconfig):
    """Train a CNN model."""
    trainloader = DataLoader(tconfig["train_data"], batch_size=tconfig["batch_size"])
    net = tconfig["model_dict"]["model"]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=tconfig["lr"])
    net.train()
    net = net.to(tconfig["device"])
    epoch_loss = 0
    epoch_acc = 0
    for _epoch in range(tconfig["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = _get_inputs_labels_from_batch(batch)
            images, labels = images.to(tconfig["device"]), labels.to(tconfig["device"])

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            images = images.cpu()
            labels = labels.cpu()
            gc.collect()
        epoch_loss /= total
        epoch_acc = correct / total
    net = net.cpu()
    gc.collect()
    return {"train_loss": epoch_loss, "train_accuracy": epoch_acc}


def _test_cnn(net, test_data, device):
    """Evaluate a CNN model using manual DataLoader loop."""
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=512, shuffle=False, num_workers=4
    )

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    net = net.to(device)
    with torch.no_grad():
        for batch in testloader:
            images, labels = _get_inputs_labels_from_batch(batch)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            images = images.cpu()
            labels = labels.cpu()

    loss /= len(testloader)
    accuracy = correct / total
    net = net.cpu()
    return {"eval_loss": loss, "eval_accuracy": accuracy}


def _test_cnn_hf_trainer(gm_dict, central_server_test_data, batch_size):
    """Evaluate a CNN model using HuggingFace Trainer API."""
    net = gm_dict["model"]
    logging.debug("Evaluating cnn model")
    testing_args = TrainingArguments(
        logging_strategy="steps",
        output_dir=".",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=batch_size,
        disable_tqdm=True,
        remove_unused_columns=False,
        report_to="none",
    )

    tester = CNNTrainer(
        model=net,
        args=testing_args,
        compute_metrics=_compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    logging.debug("length of eval dataset: %s", len(central_server_test_data))
    results = tester.evaluate(eval_dataset=central_server_test_data)
    net = net.cpu()
    return results
