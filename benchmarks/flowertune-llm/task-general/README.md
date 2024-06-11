# LLM-FlowerTune on General NLP Dataset

This directory conducts federated instruction tuning with a pretrained [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) model on [Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) dataset.
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Methodology
This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedAvg strategy.
This provides a baseline performance for the leaderboard of General NLP task.


## Environments setup
Project dependencies are defined in `pyproject.toml`. Install them with:

```shell
pip install .
```

## Experimental setup
The dataset is partitioned into 20 shards with IID fashion serving as clients.
We randomly sample 2 clients to be available for each round,
and the federated fine-tuning lasts for `200` rounds.
All settings are defined in `static_conf/config.yaml`, which is not allowed to be modified for fair competition.


## Running the task
With an activated Python environment, run the task with default config values. 
The configs are in `conf/config.yaml` and `static_conf/config.yaml`, and are loaded automatically.

```bash
python main.py
```

## Experimental results

After model fine-tuning finished, test the trained model with the commands specified in `evaluation` directory.
The expected results are shown in the table below.

 | MT-1 | MT-2 | MT-Avg |
|:----:|:----:|:------:|
| 5.8  | 4.81 |  5.34  |

Please check the `evaluation` directory for more detailed explanation.
