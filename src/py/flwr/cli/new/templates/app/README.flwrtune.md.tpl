# FlowerTune-LLM on $challenge_name Dataset

This directory conducts federated instruction tuning with a pretrained [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) model on a $challenge_name dataset.
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Methodology
This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedAvg strategy.
This provides a baseline performance for the leaderboard of $challenge_name challenge.


## Environments setup
Project dependencies are defined in `pyproject.toml`. Install them with:

```shell
pip install -e .
```

## Experimental setup
The dataset is partitioned into $num_clients shards with IID fashion serving as clients.
We randomly sample $fraction_fit clients to be available for each round,
and the federated fine-tuning lasts for `200` rounds.
All settings are defined in `$project_name/conf/static_config.yaml`, which is not allowed to be modified for fair competition.


## Running the challenge
First make sure that you have got the access to [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) model with your Hugging-Face account.
Then log in with your [User Access Token](https://huggingface.co/docs/hub/security-tokens).

```bash
huggingface-cli login --token XXXXXXX
```

With an activated Python environment, run the challenge with default config values.
The configs are in `$project_name/conf/config.yaml` and `$project_name/conf/static_config.yaml`, and are loaded automatically.

```bash
flwr run
```
