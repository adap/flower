"""Contains the train function for the target network."""

import torch
import torch.utils.data


def train(
    netw, trainloader, testloader, epochs: int, lr: float, wd: float, device, cid
) -> None:
    """Train the network on the training set."""
    net = netw
    net = net.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    # init optimizer
    optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

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
