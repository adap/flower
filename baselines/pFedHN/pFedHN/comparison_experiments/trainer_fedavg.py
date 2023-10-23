"""Contains the train function for the target network for FedAvg Algorithm."""

import torch
import torch.utils.data


# pylint: disable=too-many-arguments
def train_fedavg(
    netw: torch.nn.Module,
    trainloader,
    testloader,
    valloader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
    cid,
):
    """Train the network on the training set."""
    # pylint: disable=too-many-locals
    net = netw.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    # init optimizer
    optim = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for _i in range(epochs):
        net.train()
        optim.zero_grad()
        batch = next(iter(trainloader[int(cid)]))
        img, label = tuple(t.to(device) for t in batch)
        pred = net(img)
        loss = criteria(pred, label)
        loss.backward()
        optim.step()

        with torch.no_grad():
            net.eval()
            batch = next(iter(testloader[int(cid)]))
            img, label = tuple(t.to(device) for t in batch)
            pred = net(img)
            loss = criteria(pred, label)
            total_loss += loss.item()
            total_correct += pred.argmax(1).eq(label).sum().item()
            total_samples += label.shape[0]

    return total_loss / total_samples, total_correct / total_samples
