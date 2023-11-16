"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

import gc
import os
from typing import Callable, Dict

import flwr as fl
import torch
from flwr.common import Scalar
from omegaconf import DictConfig

from fedwav2vec2.client import SpeechBrainClient
from fedwav2vec2.models import int_model


def get_on_fit_config_fn(local_epochs: int) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {"epoch_global": str(rnd), "epochs": str(local_epochs)}
        return config

    return fit_config


def get_evaluate_fn(config: DictConfig, server_device: str, save_path: str):
    """Return function to execute during global evaluation."""
    config_ = config

    def evaluate_fn(
        server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Run centralized evaluation."""
        _ = (server_round, config)
        # int model
        asr_brain, dataset = int_model(
            config_.server_cid,
            config_,
            server_device,
            save_path,
            evaluate=True,
        )

        client = SpeechBrainClient(config_.server_cid, asr_brain, dataset)

        _, lss, err = client.evaluate_train_speech_recogniser(
            server_params=weights,
            epochs=1,
        )
        # Save model if indicated
        if config_.save_checkpoint is not None:
            if not os.path.exists(config_.save_checkpoint):
                os.mkdir(config_.save_checkpoint)
            checkpoint = os.path.join(config_.save_checkpoint, "last_checkpoint.pt")
            torch.save(asr_brain.modules.state_dict(), checkpoint)
            print(f"Checkpoint saved for round {server_round}")

        del client, asr_brain, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return lss, {"Error rate": err}

    return evaluate_fn
