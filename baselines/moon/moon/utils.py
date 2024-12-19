"""moon: A Flower Baseline."""

import numpy as np
import torch
from torch import nn


def compute_accuracy(model, dataloader, device="cpu", multiloader=False):
    """Compute accuracy."""
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch in loader:
                    x = batch["img"]
                    target = batch["label"]
                    x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                    _, _, out = model(x)
                    if len(target) == 1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.numpy()
                        )
                    else:
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.cpu().numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.cpu().numpy()
                        )
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch in dataloader:
                x = batch["img"]
                target = batch["label"]
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.cpu().numpy()
                    )
                    true_labels_list = np.append(
                        true_labels_list, target.data.cpu().numpy()
                    )
            avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss
