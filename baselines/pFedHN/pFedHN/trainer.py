"""Contains the train, test functions for the target network.

For pFedHN,pFedHNPC and FedAvg algorithms
"""


import torch
import torch.utils.data


# pylint: disable=too-many-arguments
def train(
    netw,
    trainloader,
    valloader,  # pylint: disable=unused-argument
    local_layer,
    local_optim,
    local,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
    cid,
):
    """Train the network on the training set for pFedHN/pFedHNPC."""
    # pylint: disable=too-many-locals
    net: torch.nn.Module = netw.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    # init optimizer
    optim = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    # inner updates -> obtaining theta_tilda
    net.train()
    for _i in range(epochs):
        optim.zero_grad()
        if local:
            # local optimizer for the local layer during pFedHNPC
            local_optim[int(cid)].zero_grad()

        batch = next(iter(trainloader))
        img, label = tuple(t.to(device) for t in batch)
        if local:
            # involves local layer during pFedHNPC
            net_out = net(img)
            pred = local_layer[int(cid)].to(device)(net_out)
        else:
            pred = net(img)

        loss = criteria(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
        optim.step()
        if local:
            # local optimizer step during pFedHNPC
            local_optim[int(cid)].step()

    final_state = net.state_dict()
    return final_state


def test(netw, testloader, local, local_layer, device, cid):
    """Evaluate the network on the test set for pFedHN/pFedHNPC.

    Used during evaluation metrics gathering for Federated Evaluation.
    """
    # pylint: disable=too-many-locals
    criteria = torch.nn.CrossEntropyLoss()
    running_loss, runnning_correct, running_samples = 0.0, 0.0, 0.0
    b_c = 0
    net = netw
    net.eval()
    net = net.to(device)
    with torch.no_grad():
        for batch_count, batch in enumerate(testloader):
            img, label = tuple(t.to(device) for t in batch)

            if local:
                net_out = net(img)
                pred = local_layer[int(cid)].to(device)(net_out)
            else:
                pred = net(img)

            running_loss += criteria(pred, label).item()
            runnning_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)
            b_c = batch_count

        eval_loss = running_loss / (b_c + 1)
    return eval_loss, runnning_correct, running_samples


# pylint: disable=too-many-arguments
def train_fedavg(
    netw: torch.nn.Module,
    trainloader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
):
    """Train the network on the training set for FedAvg."""
    # pylint: disable=too-many-locals
    net = netw.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    # init optimizer
    optim = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            loss = criteria(net(images), labels)
            loss.backward()
            optim.step()


def test_fedavg(net, testloader, device):
    """Evaluate the network on the test set for FedAvg.

    Used during evaluation metrics gathering for Federated Evaluation.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
