"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import gc
import os

import speechbrain as sb
import torch
from flwr.common import ndarrays_to_parameters
from hyperpyyaml import load_hyperpyyaml
from omegaconf import DictConfig

from fedwav2vec2.dataset import dataio_prepare
from fedwav2vec2.sb_recipe import ASR, get_weights


def int_model(  # pylint: disable=too-many-arguments,too-many-locals
    cid,
    config: DictConfig,
    device: str,
    save_path,
    evaluate=False,
):
    """Set up the experiment.

    Loading the hyperparameters from config files and command-line overrides, setting
    the correct path for the corresponding clients, and creating the model.
    """
    # Load hyperparameters file with command-line overrides

    if cid == 19999:
        save_path = save_path + "server"
    else:
        save_path = save_path + "/client_" + str(cid)

    # Override with FLOWER PARAMS
    if evaluate:
        overrides = {
            "output_folder": save_path,
            "number_of_epochs": 1,
            "test_batch_size": 4,
            "device": device,
            "wav2vec_output": config.huggingface_model_save_path,
        }

    else:
        overrides = {
            "output_folder": save_path,
            "wav2vec_output": config.huggingface_model_save_path,
        }

    label_path_ = config.label_path
    if label_path_ is None:
        label_path_ = os.path.join(save_path, "label_encoder.txt")

    _, run_opts, _ = sb.parse_arguments(config.sb_config)
    run_opts["device"] = device
    run_opts["data_parallel_backend"] = config.parallel_backend
    run_opts["noprogressbar"] = True  # disable tqdm progress bar

    with open(config.sb_config) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # This logic follow the data_path is a path to csv folder file
    # All train/dev/test csv files are in the same name format for server and client
    # Example:
    # server: /users/server/train.csv
    # client: /users/client_1/train.csv
    # Modify (if needed) the if else logic to fit with path format

    if int(cid) != config.server_cid:
        params["data_folder"] = os.path.join(config.data_path, "client_" + str(cid))
    else:
        params["data_folder"] = os.path.join(config.data_path, "server")

    print(f'{params["data_folder"] = }')
    params["train_csv"] = params["data_folder"] + "/ted_train.csv"
    params["valid_csv"] = params["data_folder"] + "/ted_dev.csv"
    params["test_csv"] = params["data_folder"] + "/ted_test.csv"

    if int(cid) < 1341:
        params["train_csv"] = params["data_folder"] + "/ted_train_wo5.csv"
    params["label_encoder"] = label_path_

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=config.sb_config,
        overrides=overrides,
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data, label_encoder = dataio_prepare(params)
    # Trainer initialization

    asr_brain = ASR(
        modules=params["modules"],
        hparams=params,
        run_opts=run_opts,
        checkpointer=params["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    asr_brain.label_encoder.add_unk()

    # Adding objects to trainer.
    gc.collect()
    return asr_brain, [train_data, valid_data, test_data]


def pre_trained_point(save, config: DictConfig, server_device: str):
    """Return a pre-trained model from a path and hyperparameters."""
    state_dict = torch.load(config.pre_train_model_path)

    overrides = {"output_folder": save}

    hparams = config.sb_config
    _, run_opts, _ = sb.parse_arguments(hparams)
    with open(hparams) as fin:
        params = load_hyperpyyaml(fin, overrides)

    run_opts["device"] = server_device
    run_opts["data_parallel_backend"] = config.parallel_backend
    run_opts["noprogressbar"] = True  # disable tqdm progress bar

    asr_brain = ASR(
        modules=params["modules"],
        hparams=params,
        run_opts=run_opts,
        checkpointer=params["checkpointer"],
    )

    asr_brain.modules.load_state_dict(state_dict)
    weights = get_weights(asr_brain.modules)
    pre_trained = ndarrays_to_parameters(weights)

    # Free up space after initialized
    del asr_brain, weights
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pre_trained
