"""Contains the train function for the target network."""


import torch
import torch.utils.data


# pylint: disable=too-many-arguments
def train(
    netw,
    trainloader,
    testloader,  # pylint: disable=unused-argument
    valloader,  # pylint: disable=unused-argument
    local_layers,
    local_optims,
    local,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
    cid,
):
    """Train the network on the training set."""
    # pylint: disable=too-many-locals
    net: torch.nn.Module = netw.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    # init optimizer
    optim = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    # evaluation on the sent model
    # with torch.no_grad():
    #     net.eval()
    #     batch = next(iter(testloader[int(cid)]))
    #     img, label = tuple(t.to(device) for t in batch)
    #     if local:
    #         net_out = net(img)
    #         pred = local_layers[int(cid)](net_out)
    #     else:
    #         pred = net(img)
    #     prev_loss = criteria(pred, label)
    #     prev_acc = pred.argmax(1).eq(label).sum().item() / len(label)
    #     net.train()

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
            pred = local_layers[int(cid)].to(device)(net_out)
        else:
            pred = net(img)

        loss = criteria(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
        optim.step()
        if local:
            local_optims[int(cid)].step()

    # with torch.no_grad():
    #     # net.eval()
    #     # batch = next(iter(testloader[int(cid)]))
    #     # img, label = tuple(t.to(device) for t in batch)
    #     # if local:
    #     #     net_out = net(img)
    #     #     pred = local_layers[int(cid)](net_out)
    #     # else:
    #     #     pred = net(img)
    #     # prev_loss = criteria(pred, label)
    #     # prev_acc = pred.argmax(1).eq(label).sum().item() / len(label)
    #     # net.train()
    #     running_loss = 0.0
    #     running_correct = 0.0
    #     running_samples = 0.0
    #     for batch_count,batch in enumerate(testloader[int(cid)]):
    #         img, label = tuple(t.to(device) for t in batch)
    #         if local:
    #             net_out = net(img)
    #             pred = local_layers[int(cid)](net_out)
    #         else:
    #             pred = net(img)
    #         running_loss += criteria(pred, label).item()
    #         running_correct += pred.argmax(1).eq(label).sum().item()
    #         running_samples += len(label)

    #     eval_loss = running_loss / (batch_count+1)
    #     eval_acc = running_correct / running_samples

    final_state = net.state_dict()
    # return eval_loss, eval_acc, final_state
    # return prev_loss, prev_acc, final_state
    return final_state


@torch.no_grad()
def test(
    netw,
    testloader,
    local,
    local_layers,
    cid,
    device,
):
    """Evaluate the network on the test set."""
    # pylint: disable=too-many-locals
    net = netw.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    running_loss, runnning_correct, running_samples = 0.0, 0.0, 0.0
    b_c = 0
    for batch_count, batch in enumerate(testloader[int(cid)]):
        img, label = tuple(t.to(device) for t in batch)

        # pred = net(img)
        if local:
            net_out = net(img)
            pred = local_layers[int(cid)].to(device)(net_out)
        else:
            pred = net(img)

        running_loss += criteria(pred, label).item()
        runnning_correct += pred.argmax(1).eq(label).sum().item()
        running_samples += len(label)
        b_c = batch_count

    eval_loss = running_loss / (b_c + 1)
    return eval_loss, runnning_correct, running_samples
