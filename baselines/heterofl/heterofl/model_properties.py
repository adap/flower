"""Determine number of model parameters, space it requires."""

import numpy as np
import torch
import torch.nn as nn

from heterofl.models import create_model


def get_model_properties(
    model_config, model_split_rate, model_mode, data_loader, batch_size
):
    """Calculate space occupied & number of parameters of model."""
    model_mode = model_mode.split("-") if model_mode is not None else None
    # model = create_model(model_config, model_rate=model_split_rate(i[0]))

    total_flops = 0
    total_model_parameters = 0
    ttl_prcntg = 0
    if model_mode is None:
        total_flops = _calculate_model_memory(create_model(model_config), data_loader)
        total_model_parameters = _count_parameters(create_model(model_config))
    else:
        for i in model_mode:
            total_flops += _calculate_model_memory(
                create_model(model_config, model_rate=model_split_rate[i[0]]),
                data_loader,
            ) * int(i[1])
            total_model_parameters += _count_parameters(
                create_model(model_config, model_rate=model_split_rate[i[0]])
            ) * int(i[1])
            ttl_prcntg += int(i[1])

    total_flops = total_flops / ttl_prcntg if ttl_prcntg != 0 else total_flops
    total_flops /= batch_size
    total_model_parameters = (
        total_model_parameters / ttl_prcntg
        if ttl_prcntg != 0
        else total_model_parameters
    )

    space = total_model_parameters * 32.0 / 8 / (1024**2.0)
    print("num_of_parameters = ", total_model_parameters / 1000, " K")
    print("total_flops = ", total_flops / 1000000, " M")
    print("space = ", space)

    return total_model_parameters, total_flops, space


def _calculate_model_memory(model, data_loader):
    def register_hook(module):
        def hook(module, inp, output):
            # temp = _make_flops(module, inp, output)
            # print(temp)
            for _ in module.named_parameters():
                flops.append(_make_flops(module, inp, output))

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not isinstance(module, nn.ModuleDict)
            and module != model
        ):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    flops = []
    model.apply(register_hook)

    one_dl = next(iter(data_loader))
    input_dict = {"img": one_dl[0], "label": one_dl[1]}
    with torch.no_grad():
        model(input_dict)

    for hook in hooks:
        hook.remove()

    return sum(fl for fl in flops)


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_flops(module, inp, output):
    if isinstance(inp, tuple):
        return _make_flops(module, inp[0], output)
    if isinstance(output, tuple):
        return _make_flops(module, inp, output[0])
    flops = _compute_flops(module, inp, output)
    return flops


def _compute_flops(module, inp, out):
    flops = 0
    if isinstance(module, nn.Conv2d):
        flops = _compute_conv2d_flops(module, inp, out)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        flops = np.prod(inp.shape).item()
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and module.affine:
            flops *= 2
    elif isinstance(module, nn.Linear):
        flops = np.prod(inp.size()[:-1]).item() * inp.size()[-1] * out.size()[-1]
    # else:
    #     print(f"[Flops]: {type(module).__name__} is not supported!")
    return flops


def _compute_conv2d_flops(module, inp, out):
    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups
    filters_per_channel = out_c // groups
    conv_per_position_flops = (
        module.kernel_size[0] * module.kernel_size[1] * in_c * filters_per_channel
    )
    active_elements_count = batch_size * out_h * out_w
    total_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count
    total_flops = total_conv_flops + bias_flops
    return total_flops
