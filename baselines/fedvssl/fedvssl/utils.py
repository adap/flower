"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import os
import re
import sys

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import collect_env

from fedvssl.CtP.pyvrl.apis import (  # pylint: disable=E0611,E0401
    get_root_logger,
    set_random_seed,
    train_network,
)
from fedvssl.CtP.pyvrl.builder import (  # pylint: disable=E0611,E0401
    build_dataset,
    build_model,
)


def init_p_paths(folder_name):
    """Initialise the correct working directory."""
    if os.path.basename(os.path.abspath(os.getcwd())) == f"{folder_name}":
        # if require change the current working directory.
        # The CtP folder is in the fedvssl/fedvssl.
        # Therefore, the current working directory
        # is required to be fedvssl/fedvssl
        path_dir = os.getcwd()

        os.chdir(path_dir)
        # if the CtP folder is outside the current working directory,
        # the path could be the defined as below:
        # os.chdir("..")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{path_dir}"))
    print(root_dir)
    sys.path.insert(0, os.path.join(root_dir))


def set_config_mmcv(args, cfg):
    """Set MMCV configs."""
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    cfg.gpus = args.gpus
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # if the CLI args has already specified a resume_from path,
    # we will recover training process from it.
    # otherwise, if there exists a trained model in work directory,
    # we will resume training from it

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    else:
        if os.path.isdir(cfg.work_dir):
            chk_name_list = [
                fn for fn in os.listdir(cfg.work_dir) if fn.endswith(".pth")
            ]
            # if there may exists multiple checkpoint,
            # we will select the latest one (with the highest epoch number)
            if len(chk_name_list) > 0:
                chk_epoch_list = [
                    int(re.findall(r"\d+", fn)[0])
                    for fn in chk_name_list
                    if fn.startswith("epoch")
                ]
                chk_epoch_list.sort()
                cfg.resume_from = os.path.join(
                    cfg.work_dir, f"epoch_{chk_epoch_list[-1]}.pth"
                )

    # setup data root directory
    if args.data_dir is not None:
        if "train" in cfg.data:
            cfg.data.train.data_dir = args.data_dir
        if "val" in cfg.data:
            cfg.data.val.data_dir = args.data_dir
        if "test" in cfg.data:
            cfg.data.test.data_dir = args.data_dir

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config/save the config in the working directory
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.cfg)))

    # init logger before other steps
    logger = get_root_logger(log_level=cfg.log_level)
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    return distributed, logger


def load_model(cfg):
    """Load SSL model."""
    model = build_model(cfg.model)
    # replace with videoSSL model
    return model


def load_data(cfg):
    """Load the data partition for a single client ID."""
    train_dataset = build_dataset(cfg.data.train)
    return train_dataset


def load_test_data(cfg):
    """Load test data."""
    test_dataset = build_dataset(cfg.data.val)
    return test_dataset


def train_model_cl(model, train_dataset, cfg, distributed, logger):
    """Train downstream model."""
    # model code
    train_network(model, train_dataset, cfg, distributed=distributed, logger=logger)
