from tqdm import tqdm
import torch
import random
from datasets import Dataset
import numpy as np
from collections import OrderedDict
from transformers import WhisperForConditionalGeneration

from typing import List

import flwr as fl


remove_cols = ["file", "audio", "label", "is_unknown", "speaker_id", "utterance_id"]


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
    model.eval()
    classifier.train()
    classifier.to(device)
    losses = []
    accuracies = []
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
            # print(loss, acc)
            loss_ = loss.cpu().item()
            losses.append(loss_)
            accuracies.append(acc)

            loss_avg.update(loss_)
            acc_avg.update(acc)

            t.update(data.shape[0])
            t.set_postfix(
                {"avg_loss": f"{loss_avg():.4f}", "avg_acc": f"{acc_avg():.4f}"}
            )

    return losses, accuracies


def eval_model(model, classifier, criterion, dataloader, device):
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


def prepare_silences_dataset(train_dataset, ratio_silence: float = 0.1) -> Dataset:
    # retrieve original silence audio clips
    silences = [d for d in train_dataset if d["label"] == 35]
    # figure out how many to add
    num_silence_total = int(len(train_dataset) * ratio_silence)
    # num new entries per background noise clip
    num_silence_per_bkg = num_silence_total // len(silences)

    silence_to_add = []
    for sil in silences:
        sil_array = sil["audio"]["array"]
        sr = sil["audio"]["sampling_rate"]
        print(f"Extracting audio from: {sil['file']} ...")
        for _ in range(num_silence_per_bkg):
            random_offset = random.randint(0, len(sil_array) - sr - 1)
            sil_array_crop = sil_array[random_offset : random_offset + sr]

            entry = sil
            silence_to_add.append(entry)
            silence_to_add[-1]["audio"]["array"] = sil_array_crop

    return Dataset.from_list(silence_to_add)


def construct_client_mapping(full_trainset, num_clients: int = 100):
    client_ids = list(set(full_trainset["speaker_id"]))
    client_ids.remove(
        None
    )  # remove "none" which corresponds to the _silence_ audio clips
    client_ids.sort()  # we sort this as a quick way of ensuring our client mapping is consistent between runs
    len(
        client_ids
    )  # should be 2112 (i.e. the number of participats in SpeechCommands dataset v0.02)

    # split into groups (each group represents a client)
    client_mapping = np.array_split(client_ids, num_clients)

    return client_mapping


def get_encoding_fn(processor):
    def prepare_dataset(batch):
        audio = batch["audio"]
        data = {}
        data["data"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features
        data["targets"] = (
            11
            if batch["is_unknown"]
            else (10 if batch["label"] == 35 else batch["label"])
        )
        return data

    return prepare_dataset


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_model(device, num_classes):
    # processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    encoder = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    ).get_encoder()
    encoder = encoder.to(device)
    encoder = torch.compile(encoder)

    classifier = torch.nn.Sequential(
        torch.nn.Conv1d(1500, 512, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(1),
        torch.nn.Linear(512 * 384, num_classes),
    ).to(device)
    return encoder, classifier
