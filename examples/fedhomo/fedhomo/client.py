import logging
from typing import List, Dict
import numpy as np

import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateRes, EvaluateIns,
    Parameters, GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays
)

from fedhomo.train import train_local

logging.basicConfig(level=logging.INFO)


class PlaintextClient(fl.client.Client):
    """Flower client implementing plaintext federated learning."""
    
    def __init__(
        self,
        cid: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        net: torch.nn.Module,
        epochs: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        print(f"[DEBUG] PlaintextClient {self.cid} READY", flush=True)
        logging.info(f"Initializing PlaintextClient for client {cid}")

    def get_parameters(self, config: Dict[str, str]) -> GetParametersRes:
        print(f"[DEBUG] get_parameters called for {self.cid}", flush=True)
        logging.info(f"Client {self.cid}: Getting parameters")
        try:
            ndarrays = [
                param.cpu().detach().numpy()
                for param in self.net.parameters()
            ]
            return GetParametersRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(ndarrays),
            )
        except Exception as e:
            import traceback
            logging.error(f"Client {self.cid}: FULL TRACEBACK:\n{traceback.format_exc()}")
            return GetParametersRes(
                status=Status(Code.GET_PARAMETERS_NOT_IMPLEMENTED, str(e)),
                parameters=ndarrays_to_parameters([]),
            )
        
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        logging.info(f"Client {self.cid}: Setting model parameters")
        try:
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = {
                k: torch.from_numpy(v)
                for k, v in params_dict
            }
            self.net.load_state_dict(state_dict, strict=True)
        except Exception as e:
            error_msg = f"Parameter setting failed: {str(e)}"
            logging.error(f"Client {self.cid}: {error_msg}")
            raise

    def fit(self, fit_ins: FitIns) -> FitRes:
        print(f"[DEBUG] fit called for {self.cid}", flush=True)
        logging.info(f"Client {self.cid}: Starting training")
        try:
            parameters = parameters_to_ndarrays(fit_ins.parameters)
            self.set_parameters(parameters)

            # DEBUG: controlla cosa arriva dal dataloader
            for images, labels in self.trainloader:
                logging.info(f"DEBUG images type: {type(images)}, value: {type(images[0]) if hasattr(images, '__getitem__') else 'N/A'}")
                logging.info(f"DEBUG labels type: {type(labels)}")
                break

            logging.info(f"Client {self.cid}: About to call train_local")
            
            train_local(
                self.net, self.trainloader, self.epochs,
                self.criterion, self.optimizer, torch.device("cpu")
            )
            logging.info(f"Client {self.cid}: Training completed")

            updated_params = [
                param.cpu().detach().numpy()
                for param in self.net.parameters()
            ]
            return FitRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(updated_params),
                num_examples=len(self.trainloader.dataset),
                metrics={},
            )
        except Exception as e:
            import traceback
            logging.error(f"Client {self.cid}: FULL TRACEBACK:\n{traceback.format_exc()}")
            error_msg = f"Training failed: {str(e)}"
            logging.error(f"Client {self.cid}: {error_msg}")
            return FitRes(
                status=Status(Code.FIT_NOT_IMPLEMENTED, error_msg),
                parameters=ndarrays_to_parameters([]),
                num_examples=0,
                metrics={}
            )


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[DEBUG] evaluate called for {self.cid}", flush=True)
        logging.info(f"Client {self.cid}: Evaluating model")
        try:
            loss, accuracy = self._evaluate_model()
            logging.info(f"Client {self.cid}: Evaluation loss: {loss:.4f}, accuracy: {accuracy:.4f}")
            return EvaluateRes(
                status=Status(Code.OK, "Success"),
                loss=float(loss),
                num_examples=len(self.valloader.dataset),
                metrics={"loss": loss, "accuracy": accuracy},  # ← aggiungi accuracy
            )
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logging.error(f"Client {self.cid}: {error_msg}")
            return EvaluateRes(
                status=Status(Code.EVALUATE_NOT_IMPLEMENTED, error_msg),
                loss=0.0,
                num_examples=0,
                metrics={}
            )

    def _evaluate_model(self) -> tuple[float, float]:
        self.net.eval()
        total_loss, total_samples, correct = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in self.valloader:
                inputs = batch["img"].to("cpu")
                labels = batch["label"].to("cpu")
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy
