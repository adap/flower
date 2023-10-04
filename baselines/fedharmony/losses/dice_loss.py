# Nicola Dinsdale 2020
# Dice loss for segmentation
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
########################################################################################################################

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.eps=1e-7

    def forward(self, x, target):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)





