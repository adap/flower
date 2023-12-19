"""Pre-process pre-trained SSL model for downstream fine-tuning."""

import argparse
from collections import OrderedDict

import numpy as np
import torch
from flwr.common import parameters_to_ndarrays
from mmengine.config import Config

# pylint: disable=import-error,no-name-in-module
from .CtP.pyvrl.builder import build_model


def args_parser():
    """Parse arguments to pre-process pre-trained SSL model for fine-tuning."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        default="fedvssl/conf/mmcv_conf/finetuning/r3d_18_ucf101/finetune_ucf101.py",
        type=str,
        help="Path of config file for fine-tuning.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        default="",
        type=str,
        help="Path of pre-trained SSL model.",
    )

    return parser.parse_args()


args = args_parser()

# Load config file
cfg = Config.fromfile(args.cfg_path)
cfg.model.backbone["pretrained"] = None

# Build a model using the config file
model = build_model(cfg.model)

# Conversion of the format of pre-trained SSL model from .npz files to .pth format.
params = np.load(args.pretrained_model_path, allow_pickle=True)

if params["arr_0"].shape == ():
    # For the cases where weights are stored as Parameters
    params = params["arr_0"].item()
    params = parameters_to_ndarrays(params)
else:
    # For the cases where weights are stored as NumPy arrays
    params = [np.array(v) for v in list(params["arr_0"])]

params_dict = zip(model.state_dict().keys(), params)
state_dict = {
    "state_dict": OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
}
torch.save(state_dict, "./model_pretrained.pth")
