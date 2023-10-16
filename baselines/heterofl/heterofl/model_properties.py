"""Determine number of model parameters, space it requires."""
import torch

from heterofl.models import create_model


def get_model_properties(model_config, model_split_rate, model_mode, data_loader):
    """Calculate space occupied & number of parameters of model."""
    model_mode = model_mode.split("-")
    # model = create_model(model_config, model_rate=model_split_rate(i[0]))

    total_mem_occpd = 0
    total_model_parameters = 0
    ttl_prcntg = 0
    for i in model_mode:
        total_mem_occpd += _calculate_model_memory(
            create_model(model_config, model_rate=model_split_rate[i[0]]), data_loader
        ) * int(i[1])
        total_model_parameters += _count_parameters(
            create_model(model_config, model_rate=model_split_rate[i[0]])
        ) * int(i[1])
        ttl_prcntg += int(i[1])

    total_mem_occpd /= ttl_prcntg
    total_model_parameters /= ttl_prcntg

    print("memory_occupied = ", total_mem_occpd)
    print("num_of_parameters = ", total_model_parameters)
    return total_mem_occpd, total_model_parameters


def _calculate_model_memory(model, data_loader):
    num_params = sum(p.numel() for p in model.parameters())

    def forward_hook(module, inp, output):
        activations.append(output)

    activations = []
    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    one_dl = next(iter(data_loader))
    input_dict = {"img": one_dl[0], "label": one_dl[1]}
    with torch.no_grad():
        model(input_dict)

    for hook in hooks:
        hook.remove()

    activation_memory = sum(act.numel() * 4 for act in activations)

    total_memory_bytes = num_params * 4 + activation_memory

    return total_memory_bytes / (1024**2)


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
