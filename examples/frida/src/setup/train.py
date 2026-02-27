import torch
import torch.nn.functional as F


def train(net, trainloader, device, cfg, epochs=1, proximal_mu=0.0, global_params=None):
    criterion = torch.nn.CrossEntropyLoss()

    if proximal_mu > 0.0 and global_params is not None:
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)
        params_global_flat = torch.cat(
            [torch.flatten(param) for param in global_params]
        )

    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9)
    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            if cfg.dataset == "shakespeare":
                inputs, labels = batch[cfg.text_input].to(device), batch[
                    cfg.text_label
                ].to(device)
            else:
                inputs, labels = batch[cfg.image_name].to(device), batch[
                    cfg.image_label
                ].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if proximal_mu > 0.0 and global_params is not None:
                params_localround = [p.clone().detach() for p in net.parameters()]
                params_localround_flat = torch.cat(
                    [torch.flatten(param) for param in params_localround]
                )
                proximal_term = (
                    torch.norm(params_localround_flat - params_global_flat, p=2) ** 2
                )
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()


def test(net, testloader, device, cfg):
    correct = total = 0
    loss_sum = 0.0
    net.eval()

    if cfg.dataset == "shakespeare":
        input_key = cfg.text_input
        label_key = cfg.text_label
    else:
        input_key = cfg.image_name
        label_key = cfg.image_label

    with torch.inference_mode():
        for batch in testloader:
            inputs, labels = batch[input_key].to(device), batch[label_key].to(device)
            outputs = net(inputs)
            loss_sum += F.cross_entropy(outputs, labels, reduction="sum").item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return loss_sum / total, correct / total
