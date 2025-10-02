"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

import time

time.sleep(5)
import torch
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context
from torch.utils.data import DataLoader

from whisper_example.dataset import load_data, load_data_from_disk
from whisper_example.model import (
    construct_balanced_sampler,
    get_model,
    get_params,
    set_params,
    train_one_epoch,
)

torch.set_float32_matmul_precision(
    "high"
)  #  If “high” or “medium” are set then the TensorFloat32 is used

og_threads = torch.get_num_threads()


class WhisperFlowerClient(NumPyClient):
    """A Flower client that does trains a classification head attached to the encoder of
    a Whisper-tiny encoder for Keyword spotting."""

    def __init__(
        self,
        trainset,
        batch_size: int,
        num_classes: int,
        disable_tqdm: bool,
        compile: bool,
    ):
        self.disable_tqdm = disable_tqdm
        self.batch_size = batch_size
        self.trainset = trainset.with_format("torch", columns=["data", "targets"])

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.encoder, self.classifier = get_model(self.device, num_classes, compile)

    def fit(self, parameters, config):
        """Do on-device training.

        Here the client receives the parameters of the classification head from the
        server. Then trains that classifier using the data that belongs to this client.
        Finally, The updated classifier is sent back to the server for aggregation.
        """

        # Apply the classifier parameters to the model in this client
        set_params(self.classifier, parameters)

        # construct sampler in order to have balanced batches
        sampler = None
        if len(self.trainset) > self.batch_size:
            sampler = construct_balanced_sampler(self.trainset)

        # Construct dataloader
        train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
            drop_last=True,
        )

        # Define optimizer and criterion
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)

        # Don't train if partition is very small
        run_training = len(train_loader) > 1
        metrics = {"trained": run_training}  # will be used for metrics aggregation
        if run_training:
            # Train
            avg_loss, avg_acc = train_one_epoch(
                self.encoder,
                self.classifier,
                optimizer,
                criterion,
                train_loader,
                self.device,
                disable_tqdm=self.disable_tqdm,
            )
            metrics = {**metrics, "loss": avg_loss, "accuracy": avg_acc}

        # Return local classification head and statistics
        return get_params(self.classifier), len(train_loader.dataset), metrics


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_classes = context.run_config["num-classes"]
    batch_size = context.run_config["batch-size"]
    disable_tqdm = context.run_config["disable-tqdm"]
    compile_model = context.run_config["compile-model"]

    # Some systems seem to need this, else .map stages will hang
    # Doesn't seem to be required on macOS; but it's on Ubuntu
    # even if the latter has more CPUs...
    # ! Open a PR if you know how to improve this!
    og_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    partition = load_data(
        partition_id=partition_id,
        remove_cols=context.run_config["remove-cols"],
    )

    torch.set_num_threads(og_threads)

    return WhisperFlowerClient(
        partition, batch_size, num_classes, disable_tqdm, compile_model
    ).to_client()


app = ClientApp(client_fn=client_fn)
