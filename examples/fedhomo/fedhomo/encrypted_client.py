"""fedhomo: A Flower Baseline."""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl

from fedhomo.model import get_weights, set_weights, train, test
from fedhomo.crypto import EncryptionError, DecryptionError, HomomorphicError, HomomorphicClientHandler


log = logging.getLogger(__name__)


class EncryptedFlowerClient(fl.client.Client):
    """Flower client that encrypts model parameters using CKKS homomorphic encryption."""

    def __init__(
        self,
        cid: str,
        net: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        epochs: int,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crypto_handler = HomomorphicClientHandler(cid)

    def get_parameters(self, config: Dict) -> GetParametersRes:
        """Return encrypted model parameters."""
        try:
            encrypted = self.crypto_handler.encrypt_parameters(get_weights(self.net))
            return GetParametersRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(encrypted),
            )
        except EncryptionError as e:
            log.error("Client %s: encryption failed: %s", self.cid, e)
            return GetParametersRes(
                status=Status(Code.GET_PARAMETERS_NOT_IMPLEMENTED, str(e)),
                parameters=ndarrays_to_parameters([]),
            )

    def _apply_parameters(self, raw_params, round_label: str = "") -> None:
            """Decrypt and apply parameters to the model.

            Falls back to plaintext if decryption fails (expected on round 1).
            """
            try:
                params = self.crypto_handler.process_incoming_parameters(raw_params)
                log.debug("Client %s: parameters decrypted successfully", self.cid)
            except DecryptionError:
                log.info(
                    "Client %s: received plaintext parameters (round 1 — expected behavior)",
                    self.cid,
                )
                params = raw_params

            original_shapes = [p.shape for p in self.net.parameters()]
            state_dict = {
                k: torch.from_numpy(v.reshape(shape).astype(np.float32))
                for (k, _), v, shape in zip(
                    self.net.state_dict().items(), params, original_shapes
                )
            }
            self.net.load_state_dict(state_dict, strict=True)


    def fit(self, fit_ins: FitIns) -> FitRes:
        """Train model on local data and return encrypted updated parameters."""
        try:
            self._apply_parameters(parameters_to_ndarrays(fit_ins.parameters))

            train_loss = train(self.net, self.trainloader, self.epochs, self.device)
            log.info("Client %s: train loss %.4f", self.cid, train_loss)

            encrypted = self.crypto_handler.encrypt_parameters(get_weights(self.net))
            return FitRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(encrypted),
                num_examples=len(self.trainloader.dataset),
                metrics={"train_loss": train_loss},
            )
        except HomomorphicError as e:
            log.error("Client %s: security error: %s", self.cid, e)
            return FitRes(
                status=Status(Code.FIT_NOT_IMPLEMENTED, str(e)),
                parameters=ndarrays_to_parameters([]),
                num_examples=0,
                metrics={},
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate model on local validation data."""
        try:
            self._apply_parameters(parameters_to_ndarrays(ins.parameters))
            loss, accuracy = test(self.net, self.valloader, self.device)
            log.info("Client %s: val loss %.4f, accuracy %.4f", self.cid, loss, accuracy)
            return EvaluateRes(
                status=Status(Code.OK, "Success"),
                loss=float(loss),
                num_examples=len(self.valloader.dataset),
                metrics={"accuracy": float(accuracy)},
            )
        except Exception as e:
            log.error("Client %s: evaluation failed: %s", self.cid, e)
            return EvaluateRes(
                status=Status(Code.EVALUATE_NOT_IMPLEMENTED, str(e)),
                loss=0.0,
                num_examples=0,
                metrics={},
            )
