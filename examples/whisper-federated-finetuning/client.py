import argparse
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import WhisperProcessor

from utils import (
    get_model,
    set_params,
    train_one_epoch,
    remove_cols,
    prepare_silences_dataset,
    construct_client_mapping,
    get_encoding_fn,
)

parser = argparse.ArgumentParser(description="Flower+Whisper")
parser.add_argument("--cid", type=int, required=True, help="Client id.")
parser.add_argument(
    "--server_address", type=str, required=True, help="IP of the server."
)
parser.add_argument(
    "--no-compile", action="store_true", help="To not compile client models."
)

CLIENT_DATA = "client_datasets"


class WhisperFlowerClient(fl.client.NumPyClient):
    """A Flower client that does trains a classification head attached to the encoder of
    a Whisper-tiny encoder for Keyword spotting."""

    def __init__(self, trainset, num_classes: int, disable_tqdm: bool, compile: bool):
        self.disable_tqdm = disable_tqdm
        self.trainset = trainset.with_format("torch", columns=["data", "targets"])

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.encoder, self.classifier = get_model(self.device, num_classes, compile)

    def get_parameters(self, config):
        """Return parameters in a format that is understood by the server."""
        return [val.cpu().numpy() for _, val in self.classifier.state_dict().items()]

    def fit(self, parameters, config):
        """Do on-device training.

        Here the client receives the parameters of the classification head from the
        server. Then trains that classifier using the data that belongs to this client.
        Finally, The updated classifier is sent back to the server for aggregation.
        """

        # Apply the classifier parameters to the model in this client
        set_params(self.classifier, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

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
            batch_size=batch,
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
        return self.get_parameters({}), len(train_loader.dataset), {}


def get_client_fn(
    full_data,
    encoding_fn,
    client_mapping,
    client_data_path: str = "./",
    num_classes: int = 12,
    disable_tqdm: bool = False,
    compile: bool = True,
):
    """Return a function that can be used to instantiate a particular client."""

    def client_fn(cid: str):
        torch.set_float32_matmul_precision(
            "high"
        )  #  If “high” or “medium” are set then the TensorFloat32 is used

        # if dataset hasn't been processed for this client, do so.
        # else, just load it
        try:
            full_train_dataset = load_from_disk(f"{client_data_path}/client{cid}.hf")
        except:
            # get this client's data and preprocess it
            print(f"Dataset for client {cid} not found. Pre-processing...")
            og_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            sc_client = full_data.filter(
                lambda example: example["speaker_id"] in client_mapping[int(cid)]
            )
            client_train_data = sc_client.map(
                encoding_fn, num_proc=4, remove_columns=remove_cols
            )

            # now let's add some _silence_ training examples (add 10% of total examples in this client's data)
            ratio_silences_for_client = 0.1 * (len(client_train_data) / len(full_data))
            silence_dataset = prepare_silences_dataset(
                full_data, ratio_silences_for_client
            )
            print(
                f"adding {len(silence_dataset)} to client data ({len(client_train_data)})"
            )
            silence_enc = silence_dataset.map(encoding_fn, remove_columns=remove_cols)

            full_train_dataset = concatenate_datasets([client_train_data, silence_enc])
            # save dataset. It will be loaded next time this client is spawned
            full_train_dataset.save_to_disk(f"{client_data_path}/client{cid}.hf")
            torch.set_num_threads(og_threads)

        return WhisperFlowerClient(
            full_train_dataset, num_classes, disable_tqdm, compile
        ).to_client()

    return client_fn


def main():
    """Run client."""

    # Parse input arguments
    args = parser.parse_args()

    sc_train = load_dataset("speech_commands", "v0.02", split="train", token=False)

    # generate splits
    client_mapping = construct_client_mapping(sc_train, num_clients=100)

    # pre-process all partitions (+store to disk)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    prepare_dataset_fn = get_encoding_fn(processor)

    client_fn = get_client_fn(
        sc_train,
        prepare_dataset_fn,
        client_mapping,
        compile=not (args.no_compile),
        client_data_path=CLIENT_DATA,
    )

    fl.client.start_client(
        server_address=f"{args.server_address}:8080",
        client=client_fn(args.cid),
    )


if __name__ == "__main__":
    main()
