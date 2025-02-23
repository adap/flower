import os
from datetime import datetime
from dataclasses import dataclass
from torch import nn
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_tpu_available
from transformers import TrainingArguments, IntervalStrategy, Trainer, TrainerCallback
from transformers import DataCollatorForLanguageModeling
import datasets
import evaluate
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg, FedAdam, FedAvgM
from flwr_serverless.shared_folder.in_memory_folder import InMemoryFolder
from flwr_serverless.shared_folder.local_folder import LocalFolder
from flwr_serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr_serverless.federated_node.sync_federated_node import SyncFederatedNode

# from transformers import Traniner

from experiments.dataset.tolkien_dataset_builder import TolkienDatasetBuilder

# TODO: instrument this code with flwr
# TODO: Refer to https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/integrations.py#L670
#   Implement a custom callback for HF trainer.
# os.environ["WANDB_DISABLED"] = "true"

from argparse import ArgumentParser


class TrainingSessionArgParser:
    def __init__(self):
        self.parser = ArgumentParser()
        self.add_args()

    def add_args(self):
        self.parser.add_argument(
            "--filename",
            type=str,
            default="experiments/dataset/lotr-paragraphs.json",
            help="Path to the dataset",
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="EleutherAI/gpt-neo-125M",
            help="HuggingFace CausalLM pre-trained model to be fine tuned",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=3,
            help="Number of epochs to train the model",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size to use for training",
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=2e-5,
            help="Learning rate to use for training",
        )

    def parse_args(self):
        return self.parser.parse_args()


@dataclass
class TrainingSession:
    model_name: str = "EleutherAI/gpt-neo-125M"
    # model_name: str = "EleutherAI/pythia-14M"
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    context_length: int = 128
    track: bool = False

    def __post_init__(self):
        # self.bos_token = "<|startoftext|>"
        # self.eos_token = "<|endoftext|>"
        # self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            # bos_token=self.bos_token,
            # eos_token=self.eos_token,
            # pad_token=self.pad_token,
        )

    def run(self):
        if self.track:
            import wandb

            with wandb.init(project="wikitext"):
                wandb.config.update(self.__dict__)
                self._run()
        else:
            self._run()

    def _run(self):
        self.create_datasets()
        self.create_model()
        self.create_trainer()
        self.trainer.train()

    def create_datasets(self):
        raw_datasets = datasets.load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split=["train[:100000]", "validation[:1000]"],
            # streaming=True
        )
        print("raw datasets:")
        print(raw_datasets)
        context_length = self.context_length

        def tokenize(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=False,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == context_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        tokenized_train = raw_datasets[0].map(
            tokenize, batched=True, remove_columns=raw_datasets[0].column_names
        )
        tokenized_test = raw_datasets[1].map(
            tokenize, batched=True, remove_columns=raw_datasets[1].column_names
        )

        print("tokenized:")
        print(tokenized_train)
        print("iterating:")
        for x in tokenized_train:
            print(x)
            break

        self.train_dataset = tokenized_train
        self.val_dataset = tokenized_test

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))

    def create_trainer(self):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_args = TrainingArguments(
            learning_rate=self.lr,
            output_dir=f"./results/{time}",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            evaluation_strategy="steps",
            logging_strategy="steps",
            gradient_accumulation_steps=10,
            eval_steps=50,
            logging_steps=50,
            save_strategy=IntervalStrategy.NO,
            # evaluation_strategy="epoch",
            # logging_strategy="epoch",
            report_to=["wandb"],
            eval_delay=0,
            per_device_eval_batch_size=self.batch_size,
            eval_accumulation_steps=10,
            # per_device_eval_batch_size=8,
            # logging_steps=5000,
            # logging_dir="./logs",
            # save_strategy=IntervalStrategy.NO,
            # warmup_steps=100,
            # weight_decay=0.01,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            # data_collator=lambda data: {
            #     "input_ids": torch.stack([f["input_ids"] for f in data]),
            #     "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            #     "labels": torch.stack([f["input_ids"] for f in data]),
            # },
        )


class FederatedLearningCallback(TrainerCallback):
    def __init__(self, federated_node, num_examples_per_epoch=1, **kwargs):
        super().__init__(**kwargs)
        self.node = federated_node
        self.num_examples_per_epoch = num_examples_per_epoch
        self.counted_epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Node {self.node.node_id} to begin federation at epoch end...")
        model = kwargs["model"]
        # epoch = state.epoch
        epoch = self.counted_epoch
        torch_model = model
        device = torch_model.device

        # get model weights
        node_id = self.node.node_id
        metrics = {}
        # get model weights

        model_weights = list(torch_model.cpu().parameters())
        model_weights = [w.detach().numpy() for w in model_weights]
        params: Parameters = ndarrays_to_parameters(model_weights)
        updated_params, updated_metrics = self.node.update_parameters(
            params,
            num_examples=self.num_examples_per_epoch,
            epoch=epoch,
            metrics=metrics,
        )
        self._federated_metrics = updated_metrics

        if updated_params is not None:
            # set model weights
            print("updating model weights using federation")
            updated_params = parameters_to_ndarrays(updated_params)
            model_weights = torch_model.parameters()
            for param, updated_param in zip(model_weights, updated_params):
                w = torch.from_numpy(updated_param)
                # w = w.to(device)
                param.data = nn.parameter.Parameter(w)

        torch_model.to(device)


@dataclass
class FederatedTrainingSession:
    # model_name: str = "EleutherAI/gpt-neo-125M"
    model_name: str = "EleutherAI/pythia-14M"
    num_nodes: int = 1
    context_length: int = 128
    batch_size: int = 16
    n_train_total: int = 12000  # 00
    use_async: bool = False
    track: bool = False

    def __post_init__(self):
        self.train_datasets = []
        self.test_dataset = None

    def run(self):
        if self.track:
            import wandb

            with wandb.init(project="wikitext"):
                wandb.config.update(self.__dict__)
                self._run()

        else:
            self._run()

    def _run(self):
        self.create_random_partitioned_datasets()
        self.create_models()
        self.train_concurrently()

    def create_random_partitioned_datasets(self):
        # load wikitext
        self.test_dataset = datasets.load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split="validation[:1000]",
        )

        partitioned_datasets = []
        n_train_total = self.n_train_total
        for i in range(self.num_nodes):
            start_idx = i * n_train_total // self.num_nodes
            end_idx = start_idx + n_train_total // self.num_nodes
            if i == self.num_nodes - 1:
                end_idx = n_train_total
            print(f"start={start_idx}, end={end_idx}")
            subset = datasets.load_dataset(
                "wikitext",
                "wikitext-103-v1",
                split=f"train[{start_idx}:{end_idx}]",
            )
            partitioned_datasets.append(subset)
        self.train_datasets = partitioned_datasets
        print("training datasets:")
        for i, ds in enumerate(self.train_datasets):
            print(f"{i}: {len(ds)}")
            print(ds)
        print("test dataset:")
        print(len(self.test_dataset))

    def create_models(self):
        self.federated_models = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        for i in range(self.num_nodes):
            model = AutoModelForCausalLM.from_pretrained(self.model_name).cuda()
            model.resize_token_embeddings(len(self.tokenizer))
            self.federated_models.append(model)
        return self.federated_models

    def train_concurrently(self):
        training_args = TrainingArguments(
            learning_rate=2e-5,
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=self.batch_size,
            evaluation_strategy="steps",
            logging_strategy="steps",
            gradient_accumulation_steps=10,
            eval_steps=50,
            logging_steps=50,
            save_strategy=IntervalStrategy.NO,
            # report_to=["wandb"],
            eval_delay=0,
            per_device_eval_batch_size=self.batch_size,
            eval_accumulation_steps=10,
            dataloader_drop_last=True,
        )
        trainers = []

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        accuracy_metric = evaluate.load("accuracy")
        # perplexity_metric = evaluate.load("perplexity")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return accuracy_metric.compute(predictions=preds, references=labels)

        def tokenize(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=self.context_length,
                return_overflowing_tokens=False,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == self.context_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        shared_folder = InMemoryFolder()
        # shared_folder = LocalFolder(
        #     os.path.join(os.getcwd(), "shared", str(time.time()))
        # )
        strategy = FedAvg()
        for i in range(self.num_nodes):
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            tokenized_train = self.train_datasets[i].map(
                tokenize,
                batched=True,
                # num_proc=4,
                remove_columns=self.train_datasets[i].column_names,
            )
            tokenized_test = self.test_dataset.map(
                tokenize,
                batched=True,
                # num_proc=4,
                remove_columns=self.test_dataset.column_names,
            )
            if self.use_async:
                node = AsyncFederatedNode(
                    shared_folder=shared_folder, strategy=strategy
                )
            else:
                node = SyncFederatedNode(
                    shared_folder=shared_folder,
                    strategy=strategy,
                    num_nodes=self.num_nodes,
                )

            trainer = Trainer(
                model=self.federated_models[i],
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_train,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
                if training_args.do_eval and not is_torch_tpu_available()
                else None,
                callbacks=[FederatedLearningCallback(node)],
            )
            trainers.append(trainer)

        # trainers[0].train()
        with ThreadPoolExecutor(max_workers=self.num_nodes) as executor:
            # wait for all trainers to finish
            futures = []
            for trainer in trainers:
                futures.append(executor.submit(trainer.train))
            for future in futures:
                future.result()

        # eval on test set
        print(trainers[0].model.device)
        result = trainers[0].evaluate()
        print(result)
        if self.track:
            import wandb

            wandb.log(result)
            # wandb.config.update(self.__dict__)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # model = "EleutherAI/gpt-neo-125M"
    model = "EleutherAI/pythia-14M"
    # FederatedTrainingSession(
    #     track=True, num_nodes=1, use_async=True, model_name=model
    # ).run()
    for use_async in [False]:
        for num_nodes in [2]:
            FederatedTrainingSession(
                model_name=model, track=True, num_nodes=num_nodes, use_async=use_async
            ).run()
