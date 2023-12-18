import argparse

import torch
from datasets import load_dataset
from transformers import WhisperProcessor
from torch.utils.data import DataLoader
import flwr as fl

from utils import eval_model, get_model, set_params, remove_cols, get_encoding_fn


parser = argparse.ArgumentParser(description="Flower+Whisper")
parser.add_argument("--num_rounds", type=int, default=5, help="Number of FL rounds.")
parser.add_argument(
    "--server_address", type=str, required=True, help="IP of the server."
)


NUM_CLASSES = 12
NUM_CLIENTS = 100


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # Number of local epochs done by clients
        "batch_size": 8,  # Batch size to use by clients during fit()
    }
    return config


def get_evaluate_fn(val_set, test_set, encoding_fn, num_rounds):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        """Use the entire CIFAR-10 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare model
        encoder, classifier = get_model(device, NUM_CLASSES)
        set_params(classifier, parameters)
        classifier.to(device)

        # prepare dataset
        og_threads = torch.get_num_threads()
        torch.set_num_threads(
            1
        )  # ! still, not clear to me why this is needed if we want `num_proc>1`
        if server_round == num_rounds:
            prefix = "test"
            encoded = test_set.map(encoding_fn, num_proc=4, remove_columns=remove_cols)
        else:
            prefix = "val"
            encoded = val_set.map(encoding_fn, num_proc=4, remove_columns=remove_cols)
        torch.set_num_threads(og_threads)

        val_encoded = encoded.with_format("torch", columns=["data", "targets"])
        val_loader = DataLoader(val_encoded, batch_size=64, num_workers=4)

        # Run global evaluation
        criterion = torch.nn.CrossEntropyLoss()
        loss, accuracy = eval_model(encoder, classifier, criterion, val_loader, device)

        print(f"{prefix}: --> {loss = }, {accuracy = }")

        return loss, {f"{prefix}_accuracy": accuracy}

    return evaluate


def main():
    # Parse input arguments
    args = parser.parse_args()

    # The sever will use the validation set to assess the performance of the global
    # model after each round. Then, the test set will be used for evaluating the global
    # model after the last round
    sc_val = load_dataset("speech_commands", "v0.02", split="validation", token=False)
    sc_test = load_dataset("speech_commands", "v0.02", split="test", token=False)

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    prepare_dataset_fn = get_encoding_fn(processor)

    # We use a standard FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=2,  # the strategy will wait until at least 2 clients are sampled for fit
        fraction_evaluate=0.0,  # we don't do federated evaluation in this example
        min_available_clients=2,  # the strategy will do nothing until 2 clients are connected to the server
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(
            sc_val, sc_test, prepare_dataset_fn, args.num_rounds
        ),
    )

    fl.server.start_server(
        server_address=f"{args.server_address}:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
