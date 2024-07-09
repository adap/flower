"""huggingface_example: A Flower / Hugging Face app."""

import argparse
import warnings
from collections import OrderedDict
from typing import Any

import torch
from evaluate import load as load_metric
from flwr_datasets import FederatedDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

import flwr as fl

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cpu")
CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint


def load_data(partition_id) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load IMDB data (training and eval)"""
    fds = FederatedDataset(dataset="imdb", partitioners={"train": 1_000})
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, model_max_length=512)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, trainloader, epochs) -> None:
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader) -> tuple[Any | float, Any]:
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


def main(partition_id) -> None:
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    trainloader, testloader = load_data(partition_id)

    # Flower client
    class IMDBClient(fl.client.NumPyClient):
        def get_parameters(self, config) -> list:
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters) -> None:
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config) -> tuple[list, int, dict]:
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=IMDBClient().to_client()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        choices=list(range(1_000)),
        required=True,
        type=int,
        help="Partition of the dataset divided into 1,000 iid partitions created "
        "artificially.",
    )
    partition_id = parser.parse_args().partition_id
    main(partition_id)
