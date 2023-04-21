# Copyright 2020 The Flower Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main module to run the Wav2Vec2.0 TED-LIUM 3 baseline."""


import gc
from argparse import ArgumentParser
from typing import Callable, Dict

import flwr as fl
import torch
from flwr.common import Scalar

import flwr_baselines.publications.wav2vec.tedlium.strategy as strategy
from flwr_baselines.publications.wav2vec.tedlium.client import SpeechBrainClient
from flwr_baselines.publications.wav2vec.tedlium.model import model

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
    "--running_type", type=str, default="cpu", help="running type of FL "
)
parser.add_argument(
    "--min_fit_clients", type=int, default=10, help="minimum fit clients"
)
parser.add_argument(
    "--fraction_fit",
    type=int,
    default=10,
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
    "--parallel_backend", type=bool, default=True, help="if using multi-gpus per client"
)


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
    asr_brain, dataset = model.int_model(
        19999,
        args.config_path,
        args.output,
        args.data_path,
        args.label_path,
        args.running_type,
        args.parallel_backend,
        evaluate=True,
    )

    client = SpeechBrainClient("19999", asr_brain, dataset)

    _, lss, acc = client.evaluate_train_speech_recogniser(
        server_params=weights,
        epochs=1,
    )

    del client, asr_brain, dataset
    torch.cuda.empty_cache()
    gc.collect()
    return lss, {"accuracy": acc}


if __name__ == "__main__":
    args = parser.parse_args()

    # Define resource per client
    client_resources: Dict[str, float] = {
        "num_cpus": 64.0,
        "num_gpus": 0.0,
    }

    ray_config = {"include_dashboard": False}

    if args.pre_train_model_path is not None:
        print("PRETRAINED INITIALIZE")

        PRE_TRAINED = model.pre_trained_point(
            args.pre_train_model_path,
            args.output,
            args.config_path,
            args.running_type,
            args.parallel_backend,
        )
    else:
        PRE_TRAINED = None

    strategy = strategy.CustomFedAvg(
        initial_parameters=PRE_TRAINED,
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config_fn(),
        weight_strategy=args.weight_strategy,
    )

    def client_fn(cid: str) -> fl.client.Client:
        """Function to generate the simulated clients."""
        asr_brain, dataset = model.int_model(
            cid,
            args.config_path,
            args.output,
            args.data_path,
            args.label_path,
            args.running_type,
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
