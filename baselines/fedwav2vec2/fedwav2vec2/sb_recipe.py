"""Main SpeechBrain training and testing logic."""

import gc
import time
from collections import OrderedDict
from enum import Enum, auto
from typing import Dict, Optional

import flwr as fl
import numpy as np
import speechbrain as sb
import torch
from speechbrain.dataio.dataloader import LoopedLoader
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm

# Recipe for training a sequence-to-sequence ASR system with CommonVoice.
# The system employs a wav2vec2 encoder and a CTC decoder.
# Decoding is performed with greedy decoding (will be extended to beam search).

# To run this recipe, do the following:
# > python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml

# With the default hyperparameters, the system employs a pretrained wav2vec2 encoder.
# The wav2vec2 model is pretrained following the model given in the hprams file.
# It may be dependent on the language.

# The neural network is trained with CTC on sub-word units estimated with
# Byte Pairwise Encoding (BPE).

# The experiment file is flexible enough to support a large variety of
# different systems. By properly changing the parameter files, you can try
# different encoders, decoders, tokens (e.g, characters instead of BPE),
# training languages (all CommonVoice languages), and many
# other possible variations.

# Authors
#  * Titouan Parcollet 2021


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def set_weights(weights: fl.common.NDArrays, modules, device) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict()
    valid_keys = modules.state_dict().keys()
    for key, value in zip(valid_keys, weights):
        weight = torch.Tensor(np.array(value))
        weight = weight.to(device)
        state_dict[key] = weight

    modules.load_state_dict(state_dict, strict=True)


def get_weights(modules) -> fl.common.NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    weights = []
    for _, value in modules.state_dict().items():
        weights.append(value.cpu().numpy())
    return weights


# pylint: disable=E1101,W0201,R0902
class ASR(sb.core.Brain):
    """Override of SpeechBrain default Brain class."""

    def compute_forward(self, batch, _):
        """Forward computations from the waveform batches to the output.

        probabilities.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # Forward pass
        self.feats = self.modules.wav2vec2(wavs)

        encoded_features = self.modules.enc(self.feats)
        logits = self.modules.ctc_lin(encoded_features)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute the CTC loss given predictions and targets."""
        ids = batch.id
        p_ctc, wav_lens = predictions
        chars, char_lens = batch.char_encoded

        loss = self.hparams.ctc_cost(p_ctc, chars, wav_lens, char_lens)
        sequence = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_lens, self.hparams.blank_index
        )
        # ==============================Add by Salima=======================
        # ==================================================================

        if stage != sb.Stage.TRAIN:
            self.cer_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.coer_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.cver_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.ctc_metric.append(ids, p_ctc, chars, wav_lens, char_lens)

        return loss

    def init_optimizers(self):
        """Initialize the wav2vec2 optimizer and model optimizer."""
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        stage = sb.Stage.TRAIN

        predictions = self.compute_forward(batch, stage)
        loss = self.compute_objectives(predictions, batch, stage)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec_optimizer.step()
            self.adam_optimizer.step()

        self.wav2vec_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Compute validation/test batches."""
        # Get data.
        batch = batch.to(self.device)

        predictions = self.compute_forward(batch, stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Call when a stage (either training, validation, test) starts."""
        _ = epoch
        # self.ctc_metrics = self.hparams.ctc_stats()
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.ctc_metric = self.hparams.ctc_computer()
            self.coer_metric = self.hparams.coer_computer()
            self.cver_metric = self.hparams.cver_computer()
            # self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Call at the end of a stage."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}

        # if stage == sb.Stage.TRAIN:
        #     self.train_loss = stage_loss
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            # cer = self.cer_metrics.summarize("error_rate")
            stage_stats["WER"] = self.cer_metric.summarize("error_rate")
            stage_stats["COER"] = self.coer_metric.summarize("error_rate")
            stage_stats["CVER"] = self.cver_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(self.adam_optimizer, new_lr_adam)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": old_lr_adam,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats={"loss": self.train_loss},
                valid_stats=stage_stats,
            )

            self.stage_wer = stage_stats["WER"]

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as wer_file:
                wer_file.write("CTC loss stats:\n")
                self.ctc_metric.write_stats(wer_file)
                wer_file.write("\nCER stats:\n")
                self.cer_metric.write_stats(wer_file)
                print("CTC and WER stats written to ", self.hparams.wer_file)

            self.stage_wer = stage_stats["WER"]

    def fit(  # pylint: disable=W0102,R0912,R0913,R0914,R0915
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs=Optional[Dict],
        valid_loader_kwargs=Optional[Dict],
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : Optional[Dict]
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : Optional[Dict]
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        if not isinstance(train_set, (DataLoader, LoopedLoader)):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not isinstance(
            valid_set, (DataLoader, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar
        self.modules = self.modules.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Iterate epochs
        batch_count = 0
        for epoch in epoch_counter:
            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as progress_bar:
                for batch in progress_bar:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    _, wav_lens = batch.sig
                    batch_count += wav_lens.shape[0]
                    self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                    progress_bar.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            if epoch == epoch_counter.limit:
                avg_loss = self.avg_train_loss
            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(loss, avg_valid_loss)

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)
                    valid_wer = self.stage_wer
                    if epoch == epoch_counter.limit:
                        valid_wer_last = valid_wer

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break
        if self.device == "cpu":
            self.modules = self.modules.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return batch_count, avg_loss, valid_wer_last

    def evaluate(  # pylint: disable=W0102,R0913
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs=Optional[Dict],
    ):
        """Iterate test_set and evaluate brain performance. By default, loads the best-.

        performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : Optional[Dict]
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, (DataLoader, LoopedLoader)):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.modules = self.modules.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, None)
        self.modules.eval()
        avg_test_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True, disable=not progressbar):
                self.step += 1
                _, wav_lens = batch.sig
                batch_count += wav_lens.shape[0]
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
            cer = self.stage_wer
        self.step = 0
        if self.device == "cpu":
            self.modules = self.modules.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return batch_count, avg_test_loss, cer
