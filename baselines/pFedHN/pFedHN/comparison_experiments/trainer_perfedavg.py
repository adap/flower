"""Contains the train function for the target network for PerFedAvg Algorithm."""
import copy

import torch


class PerFedAvgOptimizer(torch.optim.Optimizer):
    """PerFedAvg Optimizer."""

    def __init__(self, params, lr):
        defaults = {"lr": lr}
        super(PerFedAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        """PerFedAvgOptimizer step function."""
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                step_size = -beta if beta != 0 else -lr
                p.data.add_(other=d_p, alpha=step_size)


def train_metrics(
    model: torch.nn.Module, testloader, cid, device, lr, beta, optim, criterion
):
    """Metrics for trained network during training using testloader."""
    model.eval()
    train_num = 0
    losses = 0
    correct = 0
    test_loader = testloader[int(cid)]
    for _ in range(len(test_loader) // 2):
        x, y = next(iter(test_loader))
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optim.step()

        x, y = next(iter(test_loader))
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        output = model(x)
        loss1 = criterion(output, y)

        train_num += y.shape[0]
        losses += loss1.item()

        predicted = output.argmax(dim=1)
        correct += (predicted == y).sum().item()
    return losses / train_num, correct / train_num


def train_perfedavg(
    model: torch.nn.Module, trainloader, testloader, cid, device, lr, beta, gamma
):
    """Train the network on the training set."""
    train_loader = trainloader[int(cid)]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = PerFedAvgOptimizer(model.parameters(), lr=lr)
    learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=gamma
    )
    model.train()
    # local_epoc = 5
    local_epoc = 10
    for _ in range(local_epoc):
        for _ in range(len(train_loader) // 2):
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)

            temp_model = copy.deepcopy(list(model.parameters()))
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            for old_param, new_param in zip(model.parameters(), temp_model):
                old_param.data = new_param.data.clone()

            optimizer.step(beta=beta)
    learning_rate_scheduler.step()

    loss, acc = train_metrics(
        model, testloader, cid, device, lr, beta, optimizer, criterion
    )
    return loss, acc
