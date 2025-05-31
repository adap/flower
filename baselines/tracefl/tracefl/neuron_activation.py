"""Neuron Activation Analysis Module.

This module provides functionality for tracking and analyzing neuron activations in
neural network models during federated learning. It includes tools for capturing layer
outputs, analyzing activation patterns, and understanding neuron behavior across
different clients.
"""

import torch
import torch.nn.functional as F


def _get_all_layers_in_neural_network(net):
    """Retrieve all layers of the neural network that are either Conv2d or Linear.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model.

    Returns
    -------
    list
        A list of layers (torch.nn.Module) that are either Conv2d or Linear.
    """
    layers = []
    for layer in net.children():
        # If the layer has no submodules and is Conv2d or Linear, collect it.
        if len(list(layer.children())) == 0 and (
            isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear)
        ):
            layers.append(layer)
        # If the layer has submodules, recurse into them.
        elif len(list(layer.children())) > 0:
            layers.extend(_get_all_layers_in_neural_network(layer))
    return layers


def _get_input_and_output_of_layer(storage):
    """Get input and output of a neural network layer.

    This function creates a hook that captures both the input and output
    of a neural network layer during forward propagation.

    Args:
        storage: List to store the layer's input and output

    Returns
    -------
        Function that captures layer inputs and outputs
    """

    def hook(module, input_t, output_t):
        # Ensure exactly one input is received.
        assert (
            len(input_t) == 1
        ), f"Hook for {module.__class__.__name__} expected 1 input, got {len(input_t)}"
        storage.append(output_t.detach())

    return hook


def _insert_hooks_func(layers, storage):
    """Insert forward hooks into the specified layers.

    Parameters
    ----------
    layers : list
        List of layers (torch.nn.Module) into which hooks will be inserted.
    storage : list
        A list to collect outputs from hooks.

    Returns
    -------
    list
        A list of hook handles.
    """
    hooks = []
    for layer in layers:
        h = layer.register_forward_hook(_get_input_and_output_of_layer(storage))
        hooks.append(h)
    return hooks


def _scale_func(out, rmax=1, rmin=0):
    """Scale the output tensor to a specified range.

    Parameters
    ----------
    out : torch.Tensor
        The tensor to be scaled.
    rmax : float, optional
        Maximum value of the scaled range (default is 1).
    rmin : float, optional
        Minimum value of the scaled range (default is 0).

    Returns
    -------
    torch.Tensor
        The scaled tensor.
    """
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled


def _my_eval_neurons_activations(model, x):
    """Evaluate the neuron activations of the model for a given input.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    tuple
        A tuple containing:
          - A 1D tensor of concatenated flattened activations from all layers.
          - A list of tensors with the activations of each layer.
    """
    # Use a local list for hook outputs instead of a global variable.
    hooks_storage = []
    layer_outputs = []
    all_layers = _get_all_layers_in_neural_network(model)
    hooks = _insert_hooks_func(all_layers, hooks_storage)

    model(x)  # Forward pass; hooks will populate hooks_storage.

    for idx in range(len(all_layers)):
        outputs = F.relu(hooks_storage[idx])
        scaled_outputs = _scale_func(outputs)
        layer_outputs.append(scaled_outputs)

    # Remove hooks after the forward pass.
    for h in hooks:
        h.remove()

    return torch.cat([out.flatten() for out in layer_outputs]), layer_outputs


def get_neurons_activations(model, img):
    """Get the activations of all neurons in the model for the given input image.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    img : torch.Tensor
        The input image tensor.

    Returns
    -------
    tuple
        A tuple containing:
          - A 1D tensor of concatenated flattened activations from all layers.
          - A list of tensors with the activations of each layer.
    """
    return _my_eval_neurons_activations(model, img)
