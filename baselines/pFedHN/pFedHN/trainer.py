"""Contains the train function for the target network."""

import torch
import torch.utils.data

# pylint: disable=too-many-arguments
def train(
    netw,
    trainloader,
    testloader,
    valloader,
    local_layers,
    local_optims,
    local,
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

    # evaluation on the sent model
    with torch.no_grad():
        net.eval()
        batch = next(iter(testloader[int(cid)]))
        img, label = tuple(t.to(device) for t in batch)
        if local:
            net_out = net(img)
            pred = local_layers[int(cid)](net_out)
        else:
            pred = net(img)
        prev_loss = criteria(pred, label)
        prev_acc = pred.argmax(1).eq(label).sum().item() / len(label)
        net.train()

    # inner updates -> obtaining theta_tilda
    for _i in range(epochs):
        net.train()
        optim.zero_grad()
        if local:
            local_optims[int(cid)].zero_grad()

        batch = next(iter(trainloader[int(cid)]))
        img, label = tuple(t.to(device) for t in batch)
        if local:
            net_out = net(img)
            pred = local_layers[int(cid)](net_out)
        else:
            pred = net(img)

        loss = criteria(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
        optim.step()
        if local:
            local_optims[int(cid)].step()

    final_state = net.state_dict()
    return prev_loss,prev_acc,final_state
