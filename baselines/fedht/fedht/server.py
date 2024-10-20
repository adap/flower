from collections import OrderedDict
from fedht.model import test
from typing import Dict
import torch

# send fit round for history
def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(testloader,
                    model):   

    # global evaluation
    def evaluate(server_round, parameters, config):  # type: ignore

        # set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader)
        return loss, {"accuracy": accuracy}

    return evaluate