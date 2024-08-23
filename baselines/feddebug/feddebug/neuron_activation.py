"""The file contains the code to get the activations of all neurons."""

from operator import ne
import torch
import torch.nn.functional as F


def _get_all_layers_in_neural_network(net):
    layers = []
    for layer in net.children():
        if len(list(layer.children())) == 0 and isinstance(
            layer, (torch.nn.Conv2d, torch.nn.Linear)
        ):
            layers.append(layer)
        if len(list(layer.children())) > 0:
            temp_layers = _get_all_layers_in_neural_network(layer)
            layers = layers + temp_layers
    return layers


global HOOKS_STORAGE
HOOKS_STORAGE = []


def _get_input_and_output_of_layer(self, input_t, output_t):
    global HOOKS_STORAGE
    assert (
        len(input_t) == 1
    ), f"Hook, {self.__class__.__name__} Expected 1 input, got {len(input_t)}"
    HOOKS_STORAGE.append(output_t.detach())


def _insert_hooks_func(layers):
    all_hooks = []
    for layer in layers:
        hook = layer.register_forward_hook(_get_input_and_output_of_layer)
        all_hooks.append(hook)
    return all_hooks


# def _scale_func(out, rmax=1, rmin=0):
#     output_std = (out - out.min()) / (out.max() - out.min())
#     output_scaled = output_std * (rmax - rmin) + rmin
#     return output_scaled


def _my_eval_neurons_activations(model, x):
    global HOOKS_STORAGE
    layer2output = []
    all_layers = _get_all_layers_in_neural_network(model)

    layers = all_layers  # [1:]

    hooks = _insert_hooks_func(layers)
    model(x)  # forward pass and everthing is stored in HOOKS_STORAGE
    for l_id in range(len(layers)):
        outputs = F.relu(HOOKS_STORAGE[l_id]).cpu()
        scaled_outputs = outputs #_scale_func(outputs)
        layer2output.append(scaled_outputs)

    _ = [h.remove() for h in hooks]  # remove the hooks
    HOOKS_STORAGE = []
    neurons = torch.cat([out.flatten() for out in layer2output]).flatten().detach().cpu()
    return neurons


def get_neurons_activations(model, img):
    """Return the activations of all neurons in the model for the given input image."""
    return _my_eval_neurons_activations(model, img)
