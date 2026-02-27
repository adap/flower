import numpy as np

from .model import get_parameters, set_parameters
from .train import train


def property_inference(gradients, impact, offsets, num_of_samples, num_of_labels):
    # if impact >= 0:
    #     raise ValueError("Impact should be negative")
    labels = np.zeros(num_of_labels)
    num_of_labels = 0
    for idx, gradient in enumerate(gradients):
        if gradient < 0:
            label_count = (gradient // impact) + 1
            labels[idx] += label_count
            gradient -= label_count * impact
            num_of_labels += label_count
    gradients -= offsets

    while num_of_labels < num_of_samples:
        min_idx = np.argmin(gradients)
        labels[min_idx] += 1
        gradients[min_idx] -= impact
        num_of_labels += 1

    return labels


def get_gradients(prev_parameters, new_parameters):
    return get_weights(prev_parameters) - get_weights(new_parameters)


def get_weights(parameters):
    # Find the last (W,b) with matching out_features
    W = b = None
    for i in range(len(parameters) - 2, -1, -1):
        Wi, bi = parameters[i], parameters[i + 1]
        if getattr(Wi, "ndim", 0) == 2 and getattr(bi, "ndim", 0) == 1:
            if Wi.shape[0] == bi.shape[0]:
                W, b = Wi, bi
                break
    if W is None:  # fallback to old assumption
        W, b = parameters[-2], parameters[-1]

    if getattr(W, "ndim", 0) > 2:
        W = W.reshape(W.shape[0], -1)
    return W.sum(axis=1) + b


def get_impact(net, parameters, attackloaders, cfg, device):
    impact = 0.0
    num_of_labels = len(attackloaders)
    correction = 1.0 + 1.0 / num_of_labels
    for idx, loader in enumerate(attackloaders):
        num_of_samples = len(getattr(loader, "dataset", []))  # CHANGED
        for batch in loader:
            set_parameters(net, parameters)
            train(net, [batch], device, cfg, 1, False)
            grad = get_gradients(parameters, get_parameters(net))
            impact += grad[idx] / max(num_of_samples, 1) / num_of_labels * correction
    # For LM, gradients scale with token count; normalize once per step
    if getattr(cfg, "dataset", "") == "shakespeare":
        impact /= max(getattr(cfg, "seq_length", 1), 1)  # ADDED
    return float(impact)


def get_offsets(net, parameters, attackloaders, cfg, device):
    offsets = []
    num_of_labels = len(attackloaders)
    all_num_of_batches = sum((len(loader) for loader in attackloaders))
    for i in range(num_of_labels):
        num_of_batches = all_num_of_batches - len(attackloaders[i])
        offset = 0
        for idx, data in enumerate(attackloaders):
            if idx == i:
                continue
            for batch in data:
                set_parameters(net, parameters)
                train(net, [batch], device, cfg, 1, False)
                grad = get_gradients(parameters, get_parameters(net))
                offset += grad[i] / num_of_batches
        offsets.append(offset)
    return offsets


def old_pia(
    net, prev_params, weight_results, num_examples, attack_loaders, cfg, device
):
    impact = get_impact(net, prev_params, attack_loaders, cfg, device)
    impact = -abs(impact)

    num_clients = len(weight_results)
    num_labels = len(attack_loaders)
    if cfg.enable_offsets:
        offsets = get_offsets(net, prev_params, attack_loaders, cfg, device)
    else:
        offsets = [0] * len(attack_loaders)
    pia_results = np.full((num_clients, num_labels), 0)
    for new_params, client in weight_results:
        gradients = get_gradients(prev_params, new_params)
        attack = property_inference(
            gradients,
            impact,
            offsets,
            num_examples[client],
            len(attack_loaders),
        )
        pia_results[client] = attack
    return pia_results
