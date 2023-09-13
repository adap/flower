"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import copy
from typing import Dict, List, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import parameters_to_ndarrays


class Conv(nn.Module):
    def __init__(
        self,
        data_shape,
        hidden_size,
        classes_size,
        rate=1,
        track=False,
        norm="bn",
        scale=1,
        device="cpu",
    ):
        super().__init__()
        norm_model = norm
        scale_model = scale
        self.device = device

        if norm_model == "bn":
            norm = nn.BatchNorm2d(
                hidden_size[0], momentum=None, track_running_stats=track
            )
        elif norm_model == "in":
            norm = nn.GroupNorm(hidden_size[0], hidden_size[0])
        elif norm_model == "ln":
            norm = nn.GroupNorm(1, hidden_size[0])
        elif norm_model == "gn":
            norm = nn.GroupNorm(4, hidden_size[0])
        elif norm_model == "none":
            norm = nn.Identity()
        else:
            raise ValueError("Not valid norm")
        if scale_model:
            scaler = Scaler(rate)
        else:
            scaler = nn.Identity()
        blocks = [
            nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
            scaler,
            norm,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        for i in range(len(hidden_size) - 1):
            if norm_model == "bn":
                norm = nn.BatchNorm2d(
                    hidden_size[i + 1], momentum=None, track_running_stats=track
                )
            elif norm_model == "in":
                norm = nn.GroupNorm(hidden_size[i + 1], hidden_size[i + 1])
            elif norm_model == "ln":
                norm = nn.GroupNorm(1, hidden_size[i + 1])
            elif norm_model == "gn":
                norm = nn.GroupNorm(4, hidden_size[i + 1])
            elif norm_model == "none":
                norm = nn.Identity()
            else:
                raise ValueError("Not valid norm")
            if scale_model:
                scaler = Scaler(rate)
            else:
                scaler = nn.Identity()
            blocks.extend(
                [
                    nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                    scaler,
                    norm,
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        blocks = blocks[:-1]
        blocks.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_size[-1], classes_size),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {"loss": torch.tensor(0, device=self.device, dtype=torch.float32)}
        # x = input["img"]
        out = self.blocks(input["img"])
        if "label_split" in input:
            label_mask = torch.zeros(10, device=out.device)
            label_mask[input["label_split"]] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output["score"] = out
        output["loss"] = F.cross_entropy(out, input["label"], reduction="mean")
        return output


def conv(
    model_rate,
    data_shape,
    hidden_layers,
    classes_size,
    norm,
    global_model_rate=1,
    track=False,
):
    hidden_size = [int(np.ceil(model_rate * x)) for x in hidden_layers]
    scaler_rate = model_rate / global_model_rate
    model = Conv(data_shape, hidden_size, classes_size, scaler_rate, track , norm)
    model.apply(init_param)
    return model


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(model, train_loader, label_split, epochs, lr=0.01, momentum = 0.9 , weight_decay = 5.00e-04 ,device="cpu"):
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    model.train()
    for local_epoch in range(epochs):
        for i, input in enumerate(train_loader):
            input_dict = {}
            input_dict["img"] = input[0].to(device)
            input_dict["label"] = input[1].to(device)
            input_dict["label_split"] = torch.tensor(
                label_split, dtype=torch.int, device=device
            )
            optimizer.zero_grad()
            output = model(input_dict)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
    return


def test(model, test_loader, device="cpu"):
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(test_loader):
            input_dict = {}
            input_dict["img"] = input[0].to(device)
            input_dict["label"] = input[1].to(device)
            output = model(input_dict)
            test_loss += output["loss"].item()
            correct += (
                (output["score"].argmax(1) == input_dict["label"])
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches
    correct /= size
    return test_loss, correct


def param_model_rate_mapping(parameters, clients_model_rate, global_model_rate=1):
    unique_client_model_rate = list(set(clients_model_rate))
    print(unique_client_model_rate)

    idx_i = [None for _ in range(len(unique_client_model_rate))]
    idx = [OrderedDict() for _ in range(len(unique_client_model_rate))]
    output_weight_name = [k for k in parameters.keys() if "weight" in k][-1]
    output_bias_name = [k for k in parameters.keys() if "bias" in k][-1]
    for k, v in parameters.items():
        parameter_type = k.split(".")[-1]
        for m in range(len(unique_client_model_rate)):
            if "weight" in parameter_type or "bias" in parameter_type:
                if parameter_type == "weight":
                    if v.dim() > 1:
                        input_size = v.size(1)
                        output_size = v.size(0)
                        if idx_i[m] is None:
                            idx_i[m] = torch.arange(input_size, device=v.device)
                        input_idx_i_m = idx_i[m]
                        if k == output_weight_name:
                            output_idx_i_m = torch.arange(output_size, device=v.device)
                        else:
                            scaler_rate = (
                                unique_client_model_rate[m] / global_model_rate
                            )
                            local_output_size = int(np.ceil(output_size * scaler_rate))
                            output_idx_i_m = torch.arange(output_size, device=v.device)[
                                :local_output_size
                            ]
                        idx[m][k] = output_idx_i_m, input_idx_i_m
                        idx_i[m] = output_idx_i_m
                    else:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
                else:
                    if k == output_bias_name:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
                    else:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
            else:
                pass
    # add model rate as key to the params calculated
    param_idx_model_rate_mapping = OrderedDict()
    for i in range(len(unique_client_model_rate)):
        param_idx_model_rate_mapping[unique_client_model_rate[i]] = idx[i]

    return param_idx_model_rate_mapping


def param_idx_to_local_params(global_parameters, client_param_idx):
    local_parameters = OrderedDict()
    for k, v in global_parameters.items():
        parameter_type = k.split(".")[-1]
        if "weight" in parameter_type or "bias" in parameter_type:
            if "weight" in parameter_type:
                if v.dim() > 1:
                    local_parameters[k] = copy.deepcopy(
                        v[torch.meshgrid(client_param_idx[k])]
                    )
                else:
                    local_parameters[k] = copy.deepcopy(v[client_param_idx[k]])
            else:
                local_parameters[k] = copy.deepcopy(v[client_param_idx[k]])
        else:
            local_parameters[k] = copy.deepcopy(v)
    return local_parameters


def get_state_dict_from_param(model, parameters):
    # Load the parameters into the model
    for param_tensor, param_ndarray in zip(
        model.state_dict(), parameters_to_ndarrays(parameters)
    ):
        model.state_dict()[param_tensor].copy_(torch.from_numpy(param_ndarray))
    # Step 3: Obtain the state_dict of the model
    state_dict = model.state_dict()
    return state_dict
