"""Generate server for fedht baseline."""

from collections import OrderedDict
from typing import Dict

import torch

from fedht.model import test
from fedht.model import LogisticRegression

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
        model = LogisticRegression(cfg.num_features, cfg.num_classes)

        # set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate
