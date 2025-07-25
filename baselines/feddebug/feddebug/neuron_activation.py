"""The file contains the code to get the activations of all neurons."""

import torch
import torch.nn.functional as F


class NeuronActivation:
    """Class to get the activations of all neurons in the model."""

    def __init__(self):
        self.hooks_storage = []

    def _get_all_layers_in_neural_network(self, net):
        layers = []
        for layer in net.children():
            if len(list(layer.children())) == 0 and isinstance(
                layer, (torch.nn.Conv2d, torch.nn.Linear)
            ):
                layers.append(layer)
            if len(list(layer.children())) > 0:
                temp_layers = self._get_all_layers_in_neural_network(layer)
                layers = layers + temp_layers
        return layers

    def _get_input_and_output_of_layer(self, layer, input_t, output_t):
        assert (
            len(input_t) == 1
        ), f"Hook, {layer.__class__.__name__} Expected 1 input, got {len(input_t)}"
        self.hooks_storage.append(output_t.detach())

    def _insert_hooks(self, layers):
        all_hooks = []
        for layer in layers:
            hook = layer.register_forward_hook(self._get_input_and_output_of_layer)
            all_hooks.append(hook)
        return all_hooks

    def get_neurons_activations(self, model, img):
        """Return the activations of model for the given input."""
        layer2output = []
        layers = self._get_all_layers_in_neural_network(model)
        hooks = self._insert_hooks(layers)
        model(img)  # forward pass and everything is stored in hooks_storage
        for l_id in range(len(layers)):
            activations = F.relu(self.hooks_storage[l_id]).cpu()
            layer2output.append(activations)
        _ = [h.remove() for h in hooks]  # remove the hooks
        self.hooks_storage = []
        neurons = (
            torch.cat([out.flatten() for out in layer2output]).flatten().detach().cpu()
        )
        return neurons


def get_neurons_activations(model, img):
    """Return the activations of all neurons in the model for the given input image."""
    model = model.eval()
    neurons = NeuronActivation()
    return neurons.get_neurons_activations(model, img)
