"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from whisper_example.model import get_model, get_params, set_params, train_one_epoch
from whisper_example.task import load_data

from datasets import load_from_disk
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


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
        hist = np.histogram(self.trainset["targets"], bins=12)
        w_per_class = (
            len(self.trainset) / hist[0]
        )  # doesn't have to add up to 1 (relative is what matters)
        # print(f"{w_per_class = }")
        w_ss = [w_per_class[t] for t in self.trainset["targets"]]
        ss = WeightedRandomSampler(w_ss, len(w_ss))

        # Construct dataloader
        train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            sampler=ss,
            drop_last=True,
        )

        # Define optimizer and criterion
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.001)
        # Train
        train_one_epoch(
            self.encoder,
            self.classifier,
            optimizer,
            criterion,
            train_loader,
            self.device,
            disable_tqdm=self.disable_tqdm,
        )

        # Return local classification head and statistics
        return get_params(self.classifier), len(train_loader.dataset), {}


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    print(f"{partition_id = }")
    # is true, this `ClientApp`'s partition will be saved after being pre-processed
    save_partition = context.run_config["save-partitions-to-disk"]
    save_path = context.run_config["partitions-save-path"]
    num_classes = context.run_config["num-classes"]
    batch_size = context.run_config["batch-size"]
    disable_tqdm = context.run_config["disable-tqdm"]
    compile_model = context.run_config["compile-model"]

    torch.set_float32_matmul_precision(
        "high"
    )  #  If “high” or “medium” are set then the TensorFloat32 is used

    # If dataset hasn't been processed for this client, do so.
    # else, just load it
    try:
        partition = load_from_disk(f"{save_path}/client{partition_id}.hf")
    except:
        og_threads = torch.get_num_threads()
        partition = load_data(
            partition_id=partition_id,
            save_partition_to_disk=save_partition,
            partitions_save_path=save_path,
        )
        torch.set_num_threads(og_threads)

    return WhisperFlowerClient(
        partition, batch_size, num_classes, disable_tqdm, compile_model
    ).to_client()


app = ClientApp(client_fn=client_fn)
