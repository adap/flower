"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import random

import numpy as np
import torch
from flwr.common import NDArrays
from torch.nn import Module


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_parameters(net: Module) -> NDArrays:
    """Returns the parameters of a neural network."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
