"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

from collections import OrderedDict
from typing import List

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration


def get_model(device, num_classes, compile: bool = True):
    """Create model: Whisper-tiny Encoder + classification head."""
    encoder = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    ).get_encoder()
    encoder = encoder.to(device)
    if compile:
        encoder = torch.compile(encoder)

    # This classification head is 782K parameters
    # This is the only part of the model that is trained in federation
    classifier = torch.nn.Sequential(
        torch.nn.Conv1d(1500, 128, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(1),
        torch.nn.Linear(128 * 384, num_classes),
    ).to(device)
    return encoder, classifier


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(module: torch.nn.ModuleList):
    return [val.cpu().numpy() for _, val in module.state_dict().items()]


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.n += 1

    def __call__(self):
        return self.total / self.n


def train_one_epoch(
    model,
    classifier,
    optimizer,
    criterion,
    dataloader,
    device,
    disable_tqdm: bool = False,
):
    """Train the classification head.

    This is a very standard looking way of training PyTorch models.
    """
    model.eval()
    classifier.train()
    classifier.to(device)
    loss_avg, acc_avg = RunningAvg(), RunningAvg()
    with tqdm(total=len(dataloader.dataset), disable=disable_tqdm) as t:
        for b in dataloader:
            optimizer.zero_grad()
            data = b["data"].squeeze().to(device)
            # print(data.shape)
            labels = b["targets"].to(device)
            with torch.no_grad():
                res = model(data)[0]

            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(resres.data, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / data.shape[0]
            loss_ = loss.cpu().item()

            loss_avg.update(loss_)
            acc_avg.update(acc)

            t.update(data.shape[0])
            t.set_postfix(
                {"avg_loss": f"{loss_avg():.4f}", "avg_acc": f"{acc_avg():.4f}"}
            )


def eval_model(model, classifier, criterion, dataloader, device):
    """Evaluate a model on a validation/test set.

    This is a very normal looking way of doing this with PyTorch.
    """
    model.eval()
    classifier.eval()
    classifier.to(device)
    correct = 0
    loss_ = 0
    total = 0
    with torch.no_grad():
        for b in dataloader:
            data = b["data"].squeeze().to(device)
            # print(data.shape)
            labels = b["targets"].to(device)
            res = model(data)[0]
            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            _, predicted = torch.max(resres.data, 1)
            correct += (predicted == labels).sum().item()
            total += data.shape[0]
            loss_ += loss.cpu().item()

    accuracy = correct / total
    loss = loss_ / total

    return loss, accuracy
