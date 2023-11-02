"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import random

import numpy as np
import torch


##########################################################
# UTILS #
##########################################################
def set_seed(seed):
    """Set the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
