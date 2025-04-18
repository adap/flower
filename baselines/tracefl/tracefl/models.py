"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision
without requiring modifications) you might be better off instantiating
your  model directly from the Hydra config. In this way, swapping your
model for  another one can be done without changing the python code at
all
"""

import gc
import logging

import evaluate
import numpy as np
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)


class CNNTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for the given model and inputs.

        Parameters
        ----------
        model : torch.nn.Module
            The model to compute loss for.
        inputs : dict
            A dictionary containing the input tensors. Expected keys are "pixel_values" and "labels".
        return_outputs : bool, optional
            Whether to return the model outputs along with the loss (default is False).

        Returns
        -------
        torch.Tensor or tuple
            The computed loss if return_outputs is False; otherwise, a tuple (loss, outputs).
        """
        labels = inputs.get("labels")
        batch_inputs = inputs.get("pixel_values")
        outputs = model(batch_inputs)
        logits = outputs
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Perform a prediction step on the given inputs.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for prediction.
        inputs : dict
            A dictionary containing the input tensors.
        prediction_loss_only : bool
            If True, only the loss is returned.
        ignore_keys : list, optional
            A list of keys to ignore in the outputs (default is None).

        Returns
        -------
        tuple
            A tuple containing (loss, logits, labels). If prediction_loss_only is True, returns (loss, None, None).
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


def _compute_metrics(eval_pred):
    """Compute evaluation metrics from predictions and labels.

    Parameters
    ----------
    eval_pred : tuple
        A tuple (logits, labels) from the evaluation.

    Returns
    -------
    dict
        A dictionary containing accuracy, correct_indices, actual_labels, incorrect_indices, and predicted_labels.
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    correct_predictions = predictions == labels
    incorrect_predictions = ~correct_predictions

    correct_indices = np.where(correct_predictions)[0]
    correct_indices = torch.from_numpy(correct_indices)

    incorrect_indices = np.where(incorrect_predictions)[0]
    incorrect_indices = torch.from_numpy(incorrect_indices)

    d = {
        "accuracy": metric.compute(predictions=predictions, references=labels),
        "correct_indices": correct_indices,
        "actual_labels": labels,
        "incorrect_indices": incorrect_indices,
        "predicted_labels": predictions,
    }
    return d


def _get_inputs_labels_from_batch(batch):
    """Extract input data and labels from a batch.

    Parameters
    ----------
    batch : dict or tuple
        A batch containing the data. If a dict, expected keys are "pixel_values" and "label". Otherwise, a tuple (inputs, labels).

    Returns
    -------
    tuple
        A tuple (inputs, labels).
    """
    if "pixel_values" in batch:
        return batch["pixel_values"], batch["label"]
    else:
        x, y = batch
        return x, y


def initialize_model(name, cfg_dataset):
    """Initialize and configure the model based on its name and dataset
    configuration.

    Parameters
    ----------
    name : str
        The name or identifier of the model.
    cfg_dataset : object
        Configuration object for the dataset. Must contain attribute num_classes and channels.

    Returns
    -------
    dict
        A dictionary containing:
            - "model": The initialized model.
            - "num_classes": Number of classes in the dataset.

    Raises
    ------
    ValueError
        If the specified model name is not supported.
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

        # Set pad_token to eos_token or add a new pad_token if eos_token is not available
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
        model = None
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

        if cfg_dataset.channels == 1:
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()

    elif name == "densenet121":
        model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        if cfg_dataset.channels == 1:
            logging.info(
                "Changing the first layer of densenet model the model to accept 1 channel"
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


# ---------------------- CNN Training  ------------------------


def _train_cnn(tconfig):
    """Train a CNN model on the training dataset.

    Parameters
    ----------
    tconfig : dict
        A dictionary containing training configuration, including:
            - "train_data": Training dataset.
            - "batch_size": Batch size for training.
            - "model_dict": Dictionary with the model under the key "model".
            - "lr": Learning rate.
            - "epochs": Number of training epochs.
            - "device": Device to run training on.

    Returns
    -------
    dict
        A dictionary containing training loss ("train_loss") and training accuracy ("train_accuracy").
    """
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
            # Metrics
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
    """Evaluate a CNN model on the test dataset.

    Parameters
    ----------
    net : torch.nn.Module
        The CNN model to be evaluated.
    test_data : object
        The test dataset.
    device : str
        The device on which evaluation is performed.

    Returns
    -------
    dict
        A dictionary containing evaluation loss ("eval_loss") and evaluation accuracy ("eval_accuracy").
    """
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
    loss /= len(testloader.dataset)
    accuracy = correct / total
    net = net.cpu()
    return {"eval_loss": loss, "eval_accuracy": accuracy}


def _test_cnn_hf_trainer(gm_dict, central_server_test_data, batch_size):
    """Evaluate a CNN model using the HuggingFace Trainer on the test dataset.

    Parameters
    ----------
    gm_dict : dict
        Dictionary containing the global model under the key "model".
    central_server_test_data : object
        The test dataset.
    batch_size : int
        Batch size for evaluation.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics as returned by the trainer.
    """
    net = gm_dict["model"]
    logging.debug("Evaluating cnn model")
    testing_args = TrainingArguments(
        logging_strategy="steps",
        output_dir=".",
        do_train=False,  # Disable training
        do_eval=True,  # Enable evaluation
        per_device_eval_batch_size=batch_size,
        disable_tqdm=True,  # Enable tqdm progress bar
        remove_unused_columns=False,
        report_to="none",
    )

    tester = CNNTrainer(
        model=net,
        args=testing_args,
        # Ensure it uses the correct metrics for evaluation
        compute_metrics=_compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    logging.debug(f"lenght of eval dataset: {len(central_server_test_data)}")
    # Evaluate the model on the test dataset
    r = tester.evaluate(eval_dataset=central_server_test_data)
    net = net.cpu()
    return r


# ---------------------- Transformer Training  ------------------------


def _train_transformer(tconfig):
    """Train a transformer model on the training dataset.

    Parameters
    ----------
    tconfig : dict
        A dictionary containing training configuration, including:
            - "model_dict": Dictionary with the model under the key "model".
            - "device": Device to run training on.
            - "epochs": Number of training epochs.
            - "batch_size": Batch size for training.
            - "train_data": Training dataset.

    Returns
    -------
    dict
        A dictionary containing training loss ("train_loss") and training accuracy ("train_accuracy").
    """
    model_dict = tconfig["model_dict"]

    net = model_dict["model"]
    net = net.to(tconfig["device"])

    training_args = TrainingArguments(
        output_dir="training_output",
        lr_scheduler_type="constant",  # Set learning rate scheduler to constant
        num_train_epochs=tconfig["epochs"],
        eval_strategy="no",
        per_device_train_batch_size=tconfig["batch_size"],
        per_device_eval_batch_size=tconfig["batch_size"],
        # fp16=True,
        disable_tqdm=True,
        report_to="none",
    )

    trainer = Trainer(
        model=net,
        args=training_args,
        train_dataset=tconfig["train_data"],
        eval_dataset=tconfig["train_data"],
        compute_metrics=_compute_metrics,
    )  # type: ignore

    if "audio_feature_extractor" in model_dict:
        trainer.tokenizer = model_dict["audio_feature_extractor"]

    trainer.train()

    r = trainer.evaluate(eval_dataset=tconfig["train_data"])

    net = net.cpu()
    return {
        "train_loss": r["eval_loss"],
        "train_accuracy": r["eval_accuracy"]["accuracy"],
    }


def _test_transformer_model(args):
    """Evaluate a transformer model on the test dataset.

    Parameters
    ----------
    args : dict
        A dictionary containing:
            - "model_dict": Dictionary with the model under the key "model".
            - "test_data": The test dataset.
            - "batch_size": Batch size for evaluation.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics as returned by the trainer.
    """
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
        tester = Trainer(
            model=net,
            args=testing_args,
            compute_metrics=_compute_metrics,
            eval_dataset=central_server_test_data,
            tokenizer=model_dict["audio_feature_extractor"],
        )

    logging.debug(f"lenght of eval dataset: {len(central_server_test_data)}")

    r = tester.evaluate()
    net = net.cpu()
    return r


def global_model_eval(arch, global_net_dict, server_testdata, batch_size=32):
    """Evaluate the global model on the server test data.

    Parameters
    ----------
    arch : str
        The architecture type ("cnn" or "transformer").
    global_net_dict : dict
        Dictionary containing the global model.
    server_testdata : object
        The test dataset.
    batch_size : int, optional
        Batch size for evaluation (default is 32).

    Returns
    -------
    dict
        A dictionary with keys "loss" and "accuracy" representing evaluation metrics.
    """
    d = {}
    if arch == "cnn":
        d = _test_cnn(global_net_dict["model"], test_data=server_testdata, device="cpu")
    elif arch == "transformer":
        d = _test_transformer_model(
            {
                "model_dict": global_net_dict,
                "test_data": server_testdata,
                "batch_size": batch_size,
            }
        )

    return {
        "loss": d["eval_loss"],
        "accuracy": d["eval_accuracy"],
    }


def test_neural_network(arch, global_net_dict, server_testdata, batch_size=32):
    """Evaluate the global model on the server test data using the appropriate
    method.

    Parameters
    ----------
    arch : str
        The architecture type ("cnn" or "transformer").
    global_net_dict : dict
        Dictionary containing the global model.
    server_testdata : object
        The test dataset.
    batch_size : int, optional
        Batch size for evaluation (default is 32).

    Returns
    -------
    dict
        A dictionary with evaluation loss and accuracy.

    Raises
    ------
    ValueError
        If the specified architecture is not supported.
    """
    d = {}
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
    d["loss"] = d["eval_loss"]
    d["accuracy"] = d["eval_accuracy"]["accuracy"]

    return d


def train_neural_network(tconfig):
    """Train a neural network based on the specified architecture.

    Parameters
    ----------
    tconfig : dict
        A dictionary containing training configuration, including the key "arch" to determine the model type.

    Returns
    -------
    dict
        A dictionary containing training loss and training accuracy.

    Raises
    ------
    ValueError
        If the specified architecture is not supported.
    """
    train_dict = {}
    if tconfig["arch"] == "cnn":
        train_dict = _train_cnn(tconfig)
    elif tconfig["arch"] == "transformer":
        train_dict = _train_transformer(tconfig)
    else:
        raise ValueError(f"Architecture {tconfig['arch']} not supported")
    return train_dict


def get_parameters(model):
    """Extract and return the parameters of a PyTorch model as a list of NumPy
    arrays.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model from which to extract the parameters.

    Returns
    -------
    list of numpy.ndarray
        A list containing the model parameters converted to NumPy arrays.
    """
    model = model.cpu()
    return [val.cpu().detach().clone().numpy() for _, val in model.state_dict().items()]


def set_parameters(net, parameters):
    """Set the parameters of a PyTorch model from a list of NumPy arrays.

    Parameters
    ----------
    net : torch.nn.Module
        The PyTorch model whose parameters will be updated.
    parameters : list of numpy.ndarray
        A list of parameters (as NumPy arrays) to load into the model.

    Returns
    -------
    None
    """
    net = net.cpu()
    params_dict = zip(net.state_dict().keys(), parameters)
    new_state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    net.load_state_dict(new_state_dict, strict=False)


def create_model(model_name: str):
    if model_name == "resnet18":
        return models.resnet18(num_classes=10)
    elif model_name == "densenet121":
        return models.densenet121(num_classes=10)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
