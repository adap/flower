"""$project_name: A Flower / flwrtune app."""

from collections import OrderedDict
from typing import Callable, Dict, Tuple
from omegaconf import DictConfig
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import get_peft_model_state_dict, set_peft_model_state_dict

import flwr as fl
from flwr.common.typing import NDArrays, Scalar

from $import_name.models import get_model, cosine_annealing
from $import_name.dataset import reformat


# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        save_path,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path

        # instantiate model
        self.model = get_model(model_cfg)

        self.trainset = trainset

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = self.save_path

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()

        return (
            self.get_parameters({}),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def gen_client_fn(
    fds,
    tokenizer,
    formatting_prompts_func,
    data_collator,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    save_path: str,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Let's get the partition corresponding to the i-th client
        client_trainset = fds.load_partition(int(cid), "train")
        client_trainset = reformat(client_trainset, llm_task="$llm_challenge_str")

        return FlowerClient(
            model_cfg,
            train_cfg,
            client_trainset,
            tokenizer,
            formatting_prompts_func,
            data_collator,
            save_path,
        ).to_client()

    return client_fn

