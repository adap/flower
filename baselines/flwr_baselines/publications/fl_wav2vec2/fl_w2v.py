# Copyright 2020 The Flower Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from flwr.common.typing import Scalar
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from typing import List
from flwr.common import parameters_to_weights, Weights, weights_to_parameters
import logging
from argparse import ArgumentParser
import timeit
import flwr as fl
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.tokenizers.SentencePiece import SentencePiece
from collections import OrderedDict
from math import exp
import GPUtil
import gc
from flwr.common import parameters_to_weights, Weights, weights_to_parameters
from flwr.server.strategy.aggregate import aggregate
from typing import List
import os
from datetime import datetime
from pathlib import Path
import time
import torchaudio

from sb_w2v2 import (
    ASR,
    set_weights,
    get_weights
)
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights


parser = ArgumentParser()
parser.add_argument("--data_path", type=str, help="dataset path")
parser.add_argument('--output', type=str, default="./results/fl_fusion/", help='output folder')
parser.add_argument('--pre_train_model_path', type=str, default=None, help='path for pre-trained starting point')
parser.add_argument('--label_path', type=str, default=None,help='path for label encoder file if want to ensure the same encode for every client')
parser.add_argument('--config_path', type=str, default="./configs/", help='path to yaml file')
parser.add_argument('--running_type', type=str, default="cpu", help='running type of FL ')
parser.add_argument("--min_fit_clients", type=int, default=10, help="minimum fit clients")
parser.add_argument("--fraction_fit", type=int, default=10, help="ratio of total clients will be trained")
parser.add_argument("--min_available_clients", type=int, default=10, help="minmum available clients")
parser.add_argument("--rounds", type=int, default=30, help="global training rounds")
parser.add_argument("--local_epochs", type=int, default=5, help="local epochs on each client")
parser.add_argument("--weight_strategy", type=str, default="num", help="strategy of weighting clients in [num, loss, wer]")
parser.add_argument("--parallel_backend", type=bool, default=True, help="if using multi-gpus per client")




class SpeechBrainClient(fl.client.Client):
    def __init__(self,
        cid: str,
        asr_brain,
        dataset):

        self.cid = cid
        self.params = asr_brain.hparams
        self.modules = asr_brain.modules
        self.asr_brain = asr_brain
        self.dataset = dataset

        fl.common.logger.log(logging.DEBUG, "Starting client %s", cid)
        print("HOOOOO HEEYYYYEYEYEYYEYEYEYEYE")


    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.modules)
        parameters = fl.common.weights_to_parameters(weights)
        gc.collect()
        return ParametersRes(parameters=parameters)


    def fit(self, ins: FitIns) -> FitRes:
        print(f"==============================Client {self.cid}: fit==============================")
        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config

        # Read training configuration
        epochs = int(config["epochs"])
        global_rounds = int(config["epoch_global"])

        print("Client {} start".format(self.cid))

        (
            new_weights,
            num_examples,
            num_examples_ceil,
            fit_duration,
            avg_loss,
            avg_wer
        ) = self.train_speech_recogniser(
            weights,
            epochs,
            global_rounds=global_rounds
        )
        print(f"==============================Client {self.cid}: end==============================")
        metrics = {"train_loss": avg_loss, "wer": avg_wer}


        parameters=self.get_parameters().parameters
        del self.asr_brain.modules
        torch.cuda.empty_cache()
        gc.collect()

        return FitRes(
            parameters=parameters,
            num_examples=num_examples,
            num_examples_ceil=num_examples_ceil,
            fit_duration=fit_duration,
            metrics=metrics
        )


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # config = ins.config
        # epochs = int(config["epochs"])
        # batch_size = int(config["batch_size"])

        num_examples, loss, wer = self.train_speech_recogniser(
            server_params=weights,
            epochs=1,
            evaluate=True
        )
        torch.cuda.empty_cache()
        gc.collect()
        
        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=num_examples, loss=float(loss), accuracy=float(wer)
        )

    def train_speech_recogniser(
        self,
        server_params,
        epochs,
        evaluate=False,
        add_train=False,
        global_rounds=None
    ):
        self.params.epoch_counter.limit = epochs
        self.params.epoch_counter.current = 0

        train_data, valid_data, test_data = self.dataset
        # Set the parameters to the ones given by the server
        if server_params is not None:
            set_weights(server_params, self.modules, evaluate, add_train, self.params.device)

        # Evaluate aggerate/server model
        if evaluate:
            self.params.wer_file = self.params.output_folder + "/wer_test.txt"

            batch_count, loss, wer = self.asr_brain.evaluate(
                test_data,
                test_loader_kwargs=self.params.test_dataloader_options,
            )

            return batch_count, float(loss), float(wer)
        # Training
        fit_begin = timeit.default_timer()
        count_sample, avg_loss, avg_wer = self.asr_brain.fit(
            self.params.epoch_counter,
            train_data,
            valid_data,
            cid = self.cid,
            global_rounds = global_rounds,
            train_loader_kwargs=self.params.dataloader_options,
            valid_loader_kwargs=self.params.test_dataloader_options,
        )
        # exp operation to avg_loss and avg_wer
        avg_wer = 100 if avg_wer > 100 else avg_wer
        avg_loss = exp(- avg_loss)
        avg_wer = exp(100 - avg_wer)

        # retrieve the parameters to return
        params_list = get_weights(self.modules)

        fit_duration = timeit.default_timer() - fit_begin

        # Manage when last batch isn't full w.r.t batch size
        train_set = sb.dataio.dataloader.make_dataloader(train_data, **self.params.dataloader_options)
        if count_sample > len(train_set) * self.params.batch_size * epochs:
            count_sample = len(train_set) * self.params.batch_size * epochs

        del train_data, valid_data
        torch.cuda.empty_cache()
        gc.collect()
        return (
            params_list,
            count_sample,
            len(train_set) * self.params.batch_size * epochs,
            fit_duration,
            avg_loss,
            avg_wer
        )

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ) -> Optional[fl.common.Weights]:

        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        # Convert results
        key_name = 'train_loss' if args.weight_strategy == 'loss' else 'wer'
        weights = None

        #Define ratio merge
        if args.weight_strategy == 'num':
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            weights =  aggregate(weights_results)

        elif args.weight_strategy == 'loss' or args.weight_strategy == 'wer':
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.metrics[key_name])
                for client, fit_res in results
            ]
            weights = aggregate(weights_results)

        #Free memory for next round
        del results, weights_results
        torch.cuda.empty_cache()
        gc.collect()
        return weights_to_parameters(weights), {}


# Define custom data procedure
def dataio_prepare(hparams):

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_smaller_than"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the test data so it is faster to validate
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_smaller_than"]},
    )

    datasets = [train_data, valid_data, test_data]

    label_encoder = sb.dataio.encoder.CTCTextEncoder()


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start_seg", "end_seg")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start_seg, end_seg):
        info = torchaudio.info(wav)
        if end_seg != 0.002:
            start = int(float(start_seg) * hparams["sample_rate"])
            stop = int(float(end_seg) * hparams["sample_rate"])
            speech_segment = {"file" : wav, "start" : start, "stop" : stop}
        else:
            speech_segment = {"file": wav}
        sig = sb.dataio.dataio.read_audio(speech_segment)
        #resample to correct 16Hz if different or else remain the same
        resampled = torchaudio.transforms.Resample(
           info.sample_rate, hparams["sample_rate"],
       )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides(
        "char_list", "char_encoded"
    )
    def text_pipeline(char):
        char_list = char.strip().split()
        yield char_list
        char_encoded = label_encoder.encode_sequence_torch(char_list)
        yield char_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = hparams["label_encoder"]
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "char_encoded"],
    )
    return train_data, valid_data, test_data, label_encoder

def int_model(
    cid,
    config_path,
    save_path,
    data_path,
    label_path=None,
    device="cpu",
    parallel = True,
    evaluate=False):
    

    # Load hyperparameters file with command-line overrides
    save_path = save_path + "client_" + str(cid)
    # Override with FLOWER PARAMS
    if evaluate:
        overrides = {
            "output_folder": save_path,
            "number_of_epochs": 1,
            "test_batch_size": 4,
            "device": device,
        }

    else:
        overrides = {
            "output_folder": save_path
        }


    if label_path is None:
        label_path = os.path.join(save_path,"label_encoder.txt")

    _, run_opts, _ = sb.parse_arguments(config_path)
    run_opts["device"] = device
    run_opts["data_parallel_backend"] = parallel
    # run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': 'cpu', 'data_parallel_backend': True, 'distributed_launch': False, 'distributed_backend': 'nccl', 'find_unused_parameters': False}

    with open(config_path) as fin:
        params = load_hyperpyyaml(fin, overrides)

    '''
    This logic follow the data_path is a path to csv folder file
    All train/dev/test csv files are in the same name format for server and client
    Example: 
    server: /users/server/train.csv
    client: /users/client_1/train.csv

    Modify (if needed) the if else logic to fit with path format
    '''
    if int(cid) != 19999:
        params["data_folder"] = os.path.join(data_path,"client_"+str(cid))
    else:
        params["data_folder"] = os.path.join(data_path,"client_"+str(1300))


    params["train_csv"] = params["data_folder"] + "/ted_test.csv"
    params["valid_csv"] = params["data_folder"] + "/ted_test.csv"
    params["test_csv"] = params["data_folder"] + "/ted_test.csv"
    
    params["label_encoder"] = label_path

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=config_path,
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
    print("ASR INITITITITITIT")
    asr_brain.label_encoder = label_encoder
    asr_brain.label_encoder.add_unk()

    # Adding objects to trainer.
    gc.collect()
    return asr_brain, [train_data, valid_data, test_data]

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(args.local_epochs)
        }
        return config

    return fit_config


def evaluate(weights: fl.common.Weights):

    # int model
    asr_brain, dataset = int_model(19999,args.config_path, args.output, 
        args.data_path, args.label_path,args.running_type,args.parallel_backend, evaluate=True)

    client = SpeechBrainClient(19999, asr_brain, dataset)

    nb_ex, lss, acc = client.train_speech_recogniser(
        server_params=weights,
        epochs=1,
        evaluate = True
    )

    del client, asr_brain, dataset
    torch.cuda.empty_cache()
    gc.collect()
    return lss, {"accuracy": acc}


def pre_trained_point(path, save, hparams,device,parallel):
    state_dict = torch.load(path)
        
    overrides = {
        "output_folder": save
    }
    #start the starting point from CPU
    # run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 
    # 'device': device, 'data_parallel_backend': True, 'distributed_launch': False, 
    # 'distributed_backend': 'nccl', 'find_unused_parameters': False}

    _, run_opts, _ = sb.parse_arguments(hparams)
    with open(hparams) as fin:
        params = load_hyperpyyaml(fin, overrides)
    
    run_opts["device"] = device
    run_opts["data_parallel_backend"] = parallel

    asr_brain = ASR(
    modules=params["modules"],
    hparams=params,
    run_opts=run_opts,
    checkpointer=params["checkpointer"],
    )
    
    asr_brain.modules.load_state_dict(state_dict)
    weights = get_weights(asr_brain.modules)
    pre_trained = fl.common.weights_to_parameters(weights)

    #Free up space after initialized
    del asr_brain, weights
    gc.collect()
    torch.cuda.empty_cache()
    return pre_trained

if __name__ == "__main__":

    args = parser.parse_args()

    # Define resource per client
    client_resources = {
        "num_cpus": 8,
        "num_gpus": 1
    } 

    ray_config = {"include_dashboard": False}
    
    if args.pre_train_model_path is not None:
        print("PRETRAINED INITIALIZE")

        pre_trained = pre_trained_point(args.pre_train_model_path,args.output,args.config_path, args.running_type, args.parallel_backend)

        strategy = CustomFedAvg(
            initial_parameters = pre_trained,
            fraction_fit=args.fraction_fit,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            eval_fn=evaluate,
            on_fit_config_fn=get_on_fit_config_fn()
        )
    else:

        strategy = CustomFedAvg(
            fraction_fit=args.fraction_fit,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            eval_fn=evaluate,
            on_fit_config_fn=get_on_fit_config_fn()
        )

    # pool_size = 600

    def client_fn(cid:int):
        asr_brain, dataset = int_model(cid,args.config_path, args.output, args.data_path,
                                   args.label_path,args.running_type, args.parallel_backend)
        return SpeechBrainClient(cid, asr_brain, dataset)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.min_available_clients,
        client_resources=client_resources,
        num_rounds=args.rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )





