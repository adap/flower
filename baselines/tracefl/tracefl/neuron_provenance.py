"""Neuron provenance tracking for federated learning.

This module provides functionality for tracking and analyzing the provenance of neuron
activations and gradients across different layers of neural networks.
"""

import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import transformers
from transformers.models.bert import modeling_bert

from tracefl.models_train_eval import test_neural_network
from tracefl.utils import compute_importance


class NeuronProvenance:
    """Class for tracking neuron-level provenance in neural networks."""

    def __init__(self, cfg, arch, test_data, *, gmodel, c2model, num_classes, c2nk):
        """Initialize a NeuronProvenance instance.

        Parameters
        ----------
        cfg : object
            Configuration object.
        arch : str
            Model architecture type.
        test_data : object
            Test dataset.
        gmodel : torch.nn.Module
            Global model.
        c2model : dict
            Dictionary mapping client identifiers to their models.
        num_classes : int
            Number of classes.
        c2nk : dict
            Dictionary mapping client identifiers to number of examples.
        """
        self.arch = arch
        self.cfg = cfg
        self.test_data = test_data
        self.gmodel = gmodel
        self.c2model = c2model
        self.num_classes = num_classes
        self.device = cfg.device.device
        self.c2nk = c2nk
        self.client_ids = list(self.c2model.keys())
        self.layer_importance = compute_importance(len(get_all_layers(gmodel)))
        # Initialize attributes to avoid pylint warnings
        self.global_neurons_inputs_outputs_batch = []
        self.inputs2layer_grads = []

        logging.info("client ids: %s", self.client_ids)
        self.pk = {
            cid: self.c2nk[cid] / sum(self.c2nk.values()) for cid in self.c2nk.keys()
        }

    def _check_anomalies(self, t):
        """Check the tensor for inf or NaN values.

        Parameters
        ----------
        t : torch.Tensor
            The tensor to be checked.

        Raises
        ------
        ValueError
            If inf or NaN values are detected.
        """
        inf_mask = torch.isinf(t)
        nan_mask = torch.isnan(t)
        if inf_mask.any() or nan_mask.any():
            logging.error("Inf values: %s", torch.sum(inf_mask))
            logging.error("NaN values: %s", torch.sum(nan_mask))
            logging.error("Total values: %s", torch.numel(t))
            raise ValueError("Anomalies detected in tensor")

    def _calculate_layer_contribution(
        self,
        global_neurons_outputs: torch.Tensor,
        global_layer_grads: torch.Tensor,
        client2outputs,
        layer_id: int,
    ):
        """Calculate the contribution of each client for a specific layer.

        Parameters
        ----------
        global_neurons_outputs : torch.Tensor
            Global neurons outputs for a given input.
        global_layer_grads : torch.Tensor
            Gradients of the global layer.
        client2outputs : dict
            Dictionary mapping client IDs to their outputs.
        layer_id : int
            Identifier of the layer.

        Returns
        -------
        dict
            Dictionary mapping client IDs to their contribution for the layer.
        """
        client2avg = dict.fromkeys(self.client_ids, 0.0)
        self._check_anomalies(global_neurons_outputs)
        self._check_anomalies(global_layer_grads)
        global_layer_grads = global_layer_grads.flatten()

        for cid in self.client_ids:
            cli_acts = client2outputs[cid].to(self.device).flatten()
            self._check_anomalies(cli_acts)
            cli_part = torch.dot(cli_acts, global_layer_grads)
            client2avg[cid] = cli_part.item() * self.layer_importance[layer_id]
            cli_acts = cli_acts.cpu()

        max_contributor = max(client2avg, key=lambda cid: client2avg[cid])
        logging.debug("Max contributor: %s", max_contributor)
        return client2avg

    def _map_client_layer_contributions(self, layer_id: int):
        """Map the contributions of clients for a specific layer.

        Parameters
        ----------
        layer_id : int
            Identifier of the layer.

        Returns
        -------
        dict
            Dictionary mapping input indices to client contributions.
        """
        client2layers = {cid: get_all_layers(cm) for cid, cm in self.c2model.items()}
        global_neurons_inputs = self.global_neurons_inputs_outputs_batch[layer_id][
            0
        ].to(self.device)
        global_neurons_outputs = self.global_neurons_inputs_outputs_batch[layer_id][1]

        if isinstance(global_neurons_outputs, (tuple, list)):
            assert (
                len(global_neurons_outputs) == 1
            ), f"Expected 1 element in tuple, got {len(global_neurons_outputs)}"
            global_neurons_outputs = global_neurons_outputs[0]
        global_neurons_outputs = global_neurons_outputs.to(self.device)

        c2l = {cid: client2layers[cid][layer_id] for cid in self.client_ids}
        client2outputs = {
            c: self._evaluate_layer(layer, global_neurons_inputs)
            for c, layer in c2l.items()
        }

        input2client2contribution = {}
        for input_id in range(len(self.test_data)):
            logging.debug(
                "Mapping client contributions for %s for layer %s", input_id, layer_id
            )
            c2out_per_input = {
                cid: client2outputs[cid][input_id] for cid in self.client_ids
            }
            glayer_grads = torch.squeeze(
                self.inputs2layer_grads[input_id][layer_id][1]
            ).to(self.device)
            c2contribution = self._calculate_layer_contribution(
                global_neurons_outputs=global_neurons_outputs[input_id],
                global_layer_grads=glayer_grads,
                client2outputs=c2out_per_input,
                layer_id=layer_id,
            )
            input2client2contribution[input_id] = c2contribution

        return input2client2contribution

    def _inplace_scale_client_weights(self):
        """Scale client model weights based on the number of data points per client."""
        logging.debug("Scaling client weights based on data points per client.")
        logging.debug("Total clients in c2nk: %s", len(self.c2nk))
        logging.debug("Total clients in c2model: %s", len(self.c2model))
        for cid in self.c2model.keys():
            scale_factor = self.c2nk[cid] / sum(self.c2nk.values())
            logging.debug(
                "Scaling client %s by %s, nk = %s", cid, scale_factor, self.c2nk[cid]
            )
            with torch.no_grad():
                for cparam in self.c2model[cid].parameters():
                    cparam.data.mul_(scale_factor)
                self.c2model[cid] = self.c2model[cid].eval().cpu()

    def _capture_layer_io(self):
        """Capture the inputs and outputs of all layers in the global model."""
        hook_manager = HookManager()
        glayers = get_all_layers(self.gmodel)
        logging.debug("Total layers in global model: %s", len(glayers))
        hooks_forward = [hook_manager.insert_forward_hook(layer) for layer in glayers]
        self.gmodel.eval().to(self.device)

        test_neural_network(
            self.arch,
            {"model": self.gmodel},
            self.test_data,
            batch_size=len(self.test_data),
        )
        hook_manager.remove_hooks(hooks_forward)
        self.global_neurons_inputs_outputs_batch = hook_manager.forward_hooks_storage
        hook_manager.clear_storages()

    def _capture_layer_gradients(self):
        """Capture gradients.

        Capture the gradients for each layer of the global model for the test data.
        """
        self.inputs2layer_grads = []
        data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1)
        for m_input in data_loader:
            hook_manager = HookManager()
            set_gradients_of_model(
                self.arch, self.gmodel, m_input, self.device, hook_manager
            )
            self.inputs2layer_grads.append(hook_manager.backward_hooks_storage)
            hook_manager.clear_storages()

    def _evaluate_layer(
        self, client_layer: torch.nn.Module, global_neurons_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate a client's layer using the global model's layer inputs."""
        client_layer = client_layer.eval().to(self.device)
        outputs = client_layer(global_neurons_inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0].cpu()
        else:
            outputs = outputs.cpu()
        client_layer = client_layer.cpu()
        return outputs

    def _aggregate_client_contributions(self, input_id: int, layers2prov):
        """Aggregate the contributions.

        Aggregate the contribution of all clients for a given input across layers.
        """
        client2totalcont = dict.fromkeys(self.client_ids, 0.0)
        for lprov in layers2prov:
            for cid in self.client_ids:
                client2totalcont[cid] += lprov[input_id][cid]
        return client2totalcont

    def _normalize_contributions(self, contributions):
        """Normalize client contributions using softmax."""
        cont_tensor = torch.tensor(list(contributions.values()))
        norm = F.softmax(cont_tensor, dim=0)
        client2prov = {cid: v.item() for cid, v in zip(self.client_ids, norm)}
        return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))

    def _aggregate_input_contributions(self, layers2prov):
        """Aggregate client contributions for each input across all layers."""
        input2prov = []
        for input_id in range(len(self.test_data)):
            aggregated = self._aggregate_client_contributions(input_id, layers2prov)
            normalized = self._normalize_contributions(aggregated)
            traced_client = max(normalized, key=normalized.get)
            input2prov.append(
                {"traced_client": traced_client, "client2prov": normalized}
            )
        return input2prov

    def compute_input_provenance(self) -> List[Dict[str, Any]]:
        """Compute the provenance by aggregating client contributions.

        This method:
        1. Captures layer inputs and outputs
        2. Captures layer gradients
        3. Maps client contributions for each layer
        4. Aggregates contributions across layers for each input

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries, where each dictionary contains:
            - traced_client: ID of the client traced for this input
            - client2prov: Dictionary mapping client IDs to their contribution scores
        """
        self._capture_layer_io()
        self._capture_layer_gradients()

        layers2prov = []
        for layer_id in range(len(self.global_neurons_inputs_outputs_batch)):
            client2cont = self._map_client_layer_contributions(layer_id)
            layers2prov.append(client2cont)

        input2prov = self._aggregate_input_contributions(layers2prov)
        return input2prov

    # Alias for backward compatibility
    computeInputProvenance = compute_input_provenance


class HookManager:
    """HookManager Class."""

    def __init__(self):
        """Initialize a HookManager instance."""
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def insert_forward_hook(self, layer):
        """Insert a forward hook into the specified layer."""

        def forward_hook(_module, input_tensor, output_tensor):
            try:
                inp = input_tensor[0].detach()
            except (AttributeError, IndexError):
                inp = input_tensor[0]
            self.forward_hooks_storage.append((inp, output_tensor))

        return layer.register_forward_hook(forward_hook)

    def insert_backward_hook(self, layer):
        """Insert a backward hook into the specified layer."""

        def backward_hook(_module, input_tensor, output_tensor):
            try:
                inp = input_tensor[0].detach()
                out = output_tensor[0].detach()
            except (AttributeError, IndexError):
                inp, out = input_tensor[0], output_tensor[0]
            self.backward_hooks_storage.append((inp, out))

        return layer.register_full_backward_hook(backward_hook)

    def clear_storages(self):
        """Clear the storage for forward and backward hooks."""
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def remove_hooks(self, hooks):
        """Remove the provided hooks."""
        for hook in hooks:
            hook.remove()


def set_gradients_of_model(arch, net, input_data, device, hook_manager):
    """Set up hooks to capture gradients of the model during backward pass."""
    if arch == "transformer":
        _set_gradients_transformer_model(net, input_data, device, hook_manager)
    elif arch == "cnn":
        _set_gradients_cnn_model(net, input_data, device, hook_manager)
    else:
        raise ValueError(f"Model architecture {arch} not supported")


def _set_gradients_cnn_model(net, input_for_model, device, hook_manager):
    """Set up hooks and compute gradients for a CNN model."""
    net.zero_grad()
    all_layers = get_all_layers(net)
    hooks = [hook_manager.insert_backward_hook(layer) for layer in all_layers]
    net.to(device)
    img_input = input_for_model["pixel_values"]
    outs = net(img_input.to(device))
    logits = outs
    _, predicted = torch.max(logits, dim=1)
    logits[0, predicted].backward()
    hook_manager.remove_hooks(hooks)
    hook_manager.backward_hooks_storage.reverse()


def _set_gradients_transformer_model(net, input_data, device, hook_manager):
    """Set up hooks and compute gradients for a transformer model."""
    net.zero_grad()
    all_layers = get_all_layers(net)
    hooks = [hook_manager.insert_backward_hook(layer) for layer in all_layers]
    net.to(device)
    prepared = {
        k: torch.tensor(v, device=device).unsqueeze(0)
        for k, v in input_data.items()
        if k in ["input_ids", "token_type_ids", "attention_mask"]
    }
    outs = net(**prepared)
    logits = outs.logits
    _, predicted = torch.max(logits, dim=1)
    logits[0, predicted].backward()
    hook_manager.remove_hooks(hooks)
    hook_manager.backward_hooks_storage.reverse()


def get_all_layers(net):
    """Retrieve all layers from the model using a BERT-specific extraction."""
    return get_all_layers_bert(net)


def get_all_layers_bert(net):
    """Retrieve all layers from the model that are instances of specific layer types."""
    layers = []
    for layer in net.children():
        if isinstance(
            layer,
            (
                torch.nn.Linear,
                torch.nn.Conv2d,
                torch.nn.LayerNorm,
                transformers.pytorch_utils.Conv1D,
                modeling_bert.BertLayer,
            ),
        ):
            layers.append(layer)
        elif len(list(layer.children())) > 0:
            layers.extend(get_all_layers_bert(layer))
    return layers
