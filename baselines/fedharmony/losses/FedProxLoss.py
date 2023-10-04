# Nicola Dinsdale 2021
# CSE loss for student teacher network
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal
from losses.dice_loss import dice_loss

########################################################################################################################

class FedProxLoss(nn.Module):
    def __init__(self, global_weights, mu=0.1, main_loss='Age'):
        super(FedProxLoss, self).__init__()
        self.eps = 1e-7
        [self.global_encoder, self.global_regressor] = global_weights
        self.mu = mu
        if main_loss=='Age':
            self.main_loss = nn.MSELoss()
        elif main_loss=='Seg':
            self.main_loss = dice_loss()
        else:
            raise Exception('Mode Not Implemented')

    def forward(self, x, target, weights):
        main_loss = self.main_loss(x, target)
        diff_store = 0
        [encoder, regressor] = weights
        for key in self.global_encoder:
            val1 = self.global_encoder[key]
            val2 = encoder[key]
            diff = val1 - val2
            square = diff * diff
            diff_store += square.sum()
        for key in self.global_regressor:
            val1 = self.global_regressor[key]
            val2 = regressor[key]
            diff = val1 - val2
            square = diff * diff
            diff_store += square.sum()
        diff = torch.sqrt(diff_store)
        return main_loss + self.mu * diff
