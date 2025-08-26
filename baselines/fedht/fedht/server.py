"""Generate server for fedht baseline."""

from collections import OrderedDict
from typing import Dict
import os
import torch
import pickle
from fedht.model import test
from fedht.model import LogisticRegression
from logging import INFO
from pathlib import Path

from flwr.common import log
from flwr.server import Server

# send fit round for history
def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def gen_evaluate_fn(testloader, cfg, device: torch.device):
    """Get evaluate function for centralized metrics."""

    # global evaluation
    def evaluate(server_round, parameters, config):  # type: ignore
        """Define evaluate function for centralized metrics."""

        # define model
        model = LogisticRegression(cfg.run_config["num_features"], 
                                   cfg.run_config["num_classes"])

        # set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate

PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent

class ResultsSaverServer(Server):
    """Server to save history to disk."""

    def __init__(
        self,
        *,
        client_manager,
        strategy=None,
        results_saver_fn=None,
        context=None,
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.results_saver_fn = results_saver_fn
        self.context = context

    def fit(self, num_rounds, timeout):
        """Run federated averaging for a number of rounds."""
        history, elapsed = super().fit(num_rounds, timeout)
        if self.results_saver_fn:
            log(INFO, "Results saver function provided. Executing")
            self.results_saver_fn(history, self.context)
        return history, elapsed


def save_results_and_clean_dir(history, context):
    """Save history and clean scaler dir."""
    results = {"history": history}
    results_path = PROJECT_DIR / context.run_config["results_save_dir"] / context.run_config["agg"]
    results_path.mkdir(exist_ok=True, parents=True)
    with open(results_path / "results.pickle", "wb") as file:
        pickle.dump(results, file)
    log(INFO, f"Results saved at {file}.")
