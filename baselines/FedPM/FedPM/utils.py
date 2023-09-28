"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from .models import *
from .resnet import resnet20
import torch


class Bern(torch.autograd.Function):
    """
    Custom Bernouli function that supports gradients.
    The original Pytorch implementation of Bernouli function,
    does not support gradients.

    First-Order gradient of bernouli function with prbabilty p, is p.

    Inputs: Tensor of arbitrary shapes with bounded values in [0,1] interval
    Outputs: Randomly generated Tensor of only {0,1}, given Inputs as distributions.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
        pvals = ctx.saved_tensors
        return pvals[0] * grad_output

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def load_model(params):
    model, optimizer = None, None

    if params.get('model').get('id') == 'LeNet':
        return LeNet5()
    elif params.get('model').get('id') == 'ResNet':
        return resnet20(params)
    elif params.get('model').get('id') == 'Conv8':
        if params.get('model').get('mode') == 'mask':
            return Mask8CNN()
    elif params.get('model').get('id') == 'Conv6':
        if params.get('model').get('mode') == 'mask':
            return Mask6CNN()
        elif params.get('model').get('mode') == 'dense':
            return Dense6CNN()
    elif params.get('model').get('id') == 'Conv4':
        if params.get('model').get('mode') == 'mask':
            return Mask4CNN()
        elif params.get('model').get('mode') == 'dense':
            return Dense4CNN()
