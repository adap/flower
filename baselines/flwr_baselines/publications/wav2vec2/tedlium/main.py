"""Main module to run the Wav2Vec2.0 TED-LIUM 3 baseline."""


import gc
from argparse import ArgumentParser
from typing import Callable, Dict

import flwr as fl
import torch
from flwr.common import Scalar

from strategy import CustomFedAvg
from client import SpeechBrainClient
from model.model import int_model, pre_trained_point



def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {"epoch_global": str(rnd), "epochs": str(args.local_epochs)}
        return config

    return fit_config


def evaluate_fn(
    server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]
):
    """Function for centralized evaluation."""
    _ = (server_round, config)
    # int model
    asr_brain, dataset = int_model(
        19999,
        args.device,
        args.config_path,
        args.output,
        args.data_path,
        args.label_path,
        args.parallel_backend,
        evaluate=True,
    )

    client = SpeechBrainClient("19999", asr_brain, dataset)

    _, lss, err = client.evaluate_train_speech_recogniser(
        server_params=weights,
        epochs=1,
    )

    del client, asr_brain, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return lss, {"Error rate": err}


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/", help="dataset path")
    parser.add_argument(
        "--output", type=str, default="./docs/results/fl_fusion/", help="output folder"
    )
    parser.add_argument(
        "--pre_train_model_path",
        type=str,
        default=None,
        help="path for pre-trained starting point",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="./docs/material/label_encoder.txt",
        help="path for label encoder file if want to ensure the same encode for every client",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./docs/configs/w2v2.yaml",
        help="path to yaml file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device where simulation runs (either `cpu` or `cuda`)"
    )
    parser.add_argument(
        "--min_fit_clients", type=int, default=10, help="minimum fit clients"
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=0.01,
        help="ratio of total clients will be trained",
    )
    parser.add_argument(
        "--min_available_clients", type=int, default=10, help="minmum available clients"
    )
    parser.add_argument("--rounds", type=int, default=30, help="global training rounds")
    parser.add_argument(
        "--local_epochs", type=int, default=5, help="local epochs on each client"
    )
    parser.add_argument(
        "--weight_strategy",
        type=str,
        default="num",
        help="strategy of weighting clients in [num, loss, wer]",
    )
    parser.add_argument(
        "--parallel_backend",
        type=bool,
        default=True,
        help="if using multi-gpus per client (disable if using CPU)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    if args.device=='cpu':
        assert not(args.parallel_backend), f"If device is CPU, please set --parallel_backend to False"

    # Define resource per client
    client_resources: Dict[str, float] = {
        "num_cpus": 8.0,
        "num_gpus": 1.0,
    }

    ray_config = {"include_dashboard": False}

    if args.pre_train_model_path is not None:
        print("PRETRAINED INITIALIZE")

        pretrained = pre_trained_point(
            args.pre_train_model_path,
            args.output,
            args.config_path,
            args.device,
            args.parallel_backend,
        )
    else:
        pretrained = None

    strategy = CustomFedAvg(
        initial_parameters=pretrained,
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config_fn(),
        weight_strategy=args.weight_strategy,
    )

    def client_fn(cid: str) -> fl.client.Client:
        """Function to generate the simulated clients."""
        asr_brain, dataset = int_model(
            cid,
            args.device,
            args.config_path,
            args.output,
            args.data_path,
            args.label_path,
            args.parallel_backend,
        )
        return SpeechBrainClient(cid, asr_brain, dataset)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.min_available_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        ray_init_args=ray_config,
    )
