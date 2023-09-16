"""Contains the train function for the target network."""

import torch
import torch.utils.data


# pylint: disable=too-many-arguments
def train(
    netw,
    trainloader,
    testloader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
    cid,
) -> None:
    """Train the network on the training set."""
    # pylint: disable=too-many-locals
    net = netw.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    # init optimizer
    optim = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    # inner updates -> obtaining theta_tilda
    for _i in range(epochs):
        net.train()
        optim.zero_grad()
        batch = next(iter(trainloader[int(cid)]))
        img, label = tuple(t.to(device) for t in batch)
        pred = net(img)
        loss = criteria(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
        optim.step()

    # calculating loss and accuracy on test set for the target model
    with torch.no_grad():
        net.eval()
        batch = next(iter(testloader[int(cid)]))
        img, label = tuple(t.to(device) for t in batch)
        pred = net(img)
        loss = criteria(pred, label)
        acc = pred.argmax(1).eq(label).sum().item() / len(label)
        net.train()

    print(f"Client: {cid}, TargetModelLoss: {loss:.4f},  TargetModelAcc: {acc:.4f}")
