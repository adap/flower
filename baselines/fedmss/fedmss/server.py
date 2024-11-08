"""Generate server for fedht baseline."""

from typing import Dict

import torch

from fedmss.utils import set_model_params, get_model_parameters
from sklearn.metrics import log_loss


# send fit round for history
def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(testloader, model, cfg):
    """Get evaluate function for centralized metrics."""

    # global evaluation
    def evaluate(server_round, parameters, cfg):  # type: ignore
        """Define evaluate function for centralized metrics."""

        set_model_params(model, parameters, cfg)
        X_test, y_test = testloader
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate
