"""Define model for fedht baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader


# model code initially pulled from fedprox baseline
# generates multinomial logistic regression model via torch
class LogisticRegression(nn.Module):
    """Define LogisticRegression class."""

    def __init__(self, num_features, num_classes: int) -> None:
        """Define model."""
        super().__init__()

        # one single linear layer
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Define forward pass."""
        # forward pass; sigmoid transform included in CBELoss criterion
        output_tensor = self.linear(torch.flatten(input_tensor, 1))
        return output_tensor


# define train function that will be called by each client to train the model
def train(model, trainloader: DataLoader, cfg: DictConfig) -> None:
    """Train model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # train
    for _epoch in range(cfg.num_local_epochs):
        for _i, data in enumerate(trainloader):

            inputs, labels = data["image"], data["label"]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            model.train()

            loss = criterion(model(inputs.float()), labels.long())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()


def test(model, testloader: DataLoader) -> None:
    """Test model."""
    criterion = nn.CrossEntropyLoss()

    # initialize
    correct, total, loss = 0, 0, 0.0

    # put into evlauate mode
    model.eval()
    with torch.no_grad():
        for _i, data in enumerate(testloader):

            images, labels = data["image"], data["label"]

            outputs = model(images.float())
            total += labels.size(0)
            loss += criterion(outputs, labels.long()).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")

    loss /= len(testloader)
    accuracy = correct / total
    
    return loss, accuracy
