"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import gc

import torch
import torchvision
from torch.utils.data import DataLoader


def _get_inputs_labels_from_batch(batch):
    if "pixel_values" in batch:
        return batch["pixel_values"], batch["label"]
    else:
        x, y = batch
        return x, y


def initialize_model(name, cfg_dataset):
    """Initialize the model with the given name."""
    model_dict = {"model": None, "num_classes": cfg_dataset.num_classes}

    if name.find("resnet") != -1:
        model = None
        if "resnet18" == name:
            model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        elif "resnet34" == name:
            model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        elif "resnet50" == name:
            model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        elif "resnet101" == name:
            model = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        elif "resnet152" == name:
            model = torchvision.models.resnet152(weights="IMAGENET1K_V1")

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
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()
    elif name == "vgg16":
        model = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        if cfg_dataset.channels == 1:
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()
    else:
        raise ValueError(f"Model {name} not supported")

    return model_dict


def _train(tconfig):
    """Train the network on the training set."""
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
            del images
            del labels
            gc.collect()
        epoch_loss /= total
        epoch_acc = correct / total
    net = net.cpu()
    del net
    gc.collect()
    return {"train_loss": epoch_loss, "train_accuracy": epoch_acc}


def global_model_eval(arch, global_net_dict, server_testdata, batch_size=64):
    """Evaluate the global model on the server test data."""
    d = {}
    if arch == "cnn":
        # d = _test_cnn_pl_trianer(
        #     global_net_dict,
        #     central_server_test_data=server_testdata,
        #     batch_size=batch_size,
        # )[0]
        d = test(global_net_dict["model"], test_data=server_testdata, device="cuda")
    return {
        "loss": d["eval_loss"],
        "accuracy": d["eval_accuracy"],
    }


def test(net, test_data, device):
    """Evaluate the network on the entire test set."""
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
            del images
            del labels
    loss /= len(testloader.dataset)
    accuracy = correct / total
    net = net.cpu()
    del net
    return {"eval_loss": loss, "eval_accuracy": accuracy}


def train_neural_network(tconfig):
    """Train the neural network."""
    train_dict = _train(tconfig)
    return train_dict
