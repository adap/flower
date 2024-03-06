from collections import OrderedDict

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


def get_model():
    """Return a pretrained ViT with all layers frozen except output head."""

    # Instantiate a pre-trained ViT-B on ImageNet
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # We're going to federated the finetuning of this model
    # using the Oxford Flowers-102 dataset. One easy way to achieve
    # this is by re-initializing the output block of the ViT so it
    # outputs 102 clases instead of the default 1k
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, 102)

    # Disable gradients for everything
    model.requires_grad_(False)
    # Now enable just for output head
    model.heads.requires_grad_(True)

    return model


def set_parameters(model, parameters):
    """Apply the parameters to the model.

    Recall this example only federates the head of the ViT so that's the only part of
    the model we need to load.
    """
    finetune_layers = model.heads
    params_dict = zip(finetune_layers.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    finetune_layers.load_state_dict(state_dict, strict=True)


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    # A very standard training loop for image classification
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    return avg_loss / len(trainloader)


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
