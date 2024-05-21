import argparse

import torch
from datasets import load_dataset
from transformers import WhisperProcessor

import flwr as fl

from client import get_client_fn
from server import fit_config, get_evaluate_fn
from utils import construct_client_mapping, get_encoding_fn

parser = argparse.ArgumentParser(description="Flower+Whisper")

parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds.")
parser.add_argument(
    "--num_cpus", type=int, default=4, help="Number of CPUs reserved for each client."
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.5,
    help="GPU ratio reserved for each client (`num_gpus`=1.0 means one client gets the whole GPU)",
)
parser.add_argument(
    "--preprocess",
    action="store_true",
    help="Preprocesses all client's datasets and exits (creates ~83GB data)",
)

NUM_CLASSES = 12
NUM_CLIENTS = 100
CLIENT_DATA = "client_datasets"
torch.set_float32_matmul_precision(
    "high"
)  #  If “high” or “medium” are set then the TensorFloat32 is used


def main():
    # Parse input arguments
    args = parser.parse_args()

    # dataset download and preparation
    sc_train = load_dataset("speech_commands", "v0.02", split="train", token=False)
    sc_val = load_dataset("speech_commands", "v0.02", split="validation", token=False)
    sc_test = load_dataset("speech_commands", "v0.02", split="test", token=False)

    # generate splits
    client_mapping = construct_client_mapping(sc_train, num_clients=NUM_CLIENTS)

    # pre-process all partitions (+store to disk)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    prepare_dataset_fn = get_encoding_fn(processor)
    if args.preprocess:
        import sys

        client_fn = get_client_fn(
            sc_train, prepare_dataset_fn, client_mapping, CLIENT_DATA, NUM_CLASSES
        )

        for i in range(NUM_CLIENTS):
            _ = client_fn(str(i))
        print("Preprocessing completed. Run the code again without `--preprocess`")
        sys.exit(0)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=10,
        fraction_evaluate=0.0,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(
            sc_val, sc_test, prepare_dataset_fn, args.num_rounds
        ),
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(
            sc_train,
            prepare_dataset_fn,
            client_mapping,
            CLIENT_DATA,
            NUM_CLASSES,
            disable_tqdm=True,
        ),
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": args.num_cpus, "num_gpus": args.num_gpus},
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
